# 적용한 모든 방법 — 종합 정리 (Master)

작성 2026-06-29. 이 문서는 `simulation/` 프로젝트에서 **지금까지 적용한 모든 방법**을
방법별 동기·구현·수식·결과·교훈·코드위치로 정리한 단일 레퍼런스다. 새 창/세미나/논문
작업의 기반으로 쓰기 위한 것. 각 항목의 상세는 끝에 링크한 개별 `*_RESULTS.md` 참조.

---

## 0. 공통 셋업 (모든 실험 고정)

- **데이터**: 배터리/전류 4채널 시계열(VBAT, IBAT, ICHG, ILOAD), 10Hz.
- **task**: 동작/상태 분류. `--merge11`로 클래스 11→0 병합 → **11 클래스**.
- **데이터량(소량)**: 클래스당 첫 600행(=1분)만 사용(`--samples_per_label 600`).
- **표준 split**: `--first_per_label --window 10 --stride 4 --split_mode temporal
  --trim_head 10 --trim_tail 30` → **train 1320 / val 144 / test 144** (temporal, 누수 없음).
- **평가 규칙**: val/test는 **real만**. 합성/증강 데이터 절대 미포함.
- **seed**: 반드시 **3-seed(42/1/7) 평균**. test 144 윈도우라 단일 seed ±0.014는 노이즈.
- **메트릭**: accuracy, macro precision/recall, **weighted F1**(`metrics.py` 기준).
- **env**: 데이터 코드(pandas) = `/home/seop/.conda/envs/open-mmlab/bin/python` (torch 1.11).
- **핵심 코드**: `train.py`(학습 루프/KD/SDFT), `model.py`(아키텍처), `dataset.py`,
  `diffusion.py`/`train_diffusion.py`(생성기), `train_with_synth.py`(합성 증강 학습),
  `closed_loop.py`(표적 증강), `select_synthetic.py`(필터), `noise_augment.py`(노이즈).

### 한눈 결과 보드 (3-seed 평균, weighted F1 기준)

| 단계 / 방법 | acc | f1 | 비고 |
|---|---:|---:|---|
| vanilla PatchTST (temporal, no-freq) | 0.815 | 0.812 | 패치 2개, 트랜스포머 무력화 |
| overlap PatchTST (no-freq) | 0.847 | 0.843 | 패치만 늘림 |
| CNN-BiLSTM (100ep) | 0.826 | 0.817 | |
| TimesNet (100ep) | 0.894 | 0.890 | |
| ResNet1D (100ep) | 0.917 | 0.915 | 강한 CNN 베이스 |
| **hpatchfix base** (overlap+FFT, 50ep) | **0.940** | **0.938** | backbone 개선 핵심 |
| + hardaug (무차별 합성) | 0.926 | 0.923 | **하락** |
| + sd_scratch (teacher-soft KD) | 0.949 | 0.948 | baseline 초과 |
| + **sd_init** (teacher init + KD) | **0.958** | **0.958** | 최고, teacher 동급 |
| + ca_init (confusion-aware KD) | 0.956 | 0.955 | sd_init과 유사 |
| online SDFT light (EMA consistency) | 0.963 | 0.962 | preliminary, noise 효과 큼 |
| **closed-loop targeted NF** (hcb 기준) | **0.914** | — | baseline +0.030, **가장 일관** |
| (참고) Utica frozen foundation + probe | 0.958 | 0.957 | 사전학습 대비 동급 |
| freq branch = PatchTST (vs CNN) | 0.901 | 0.898 | **하락** (negative) |

> 주의: closed-loop은 `hybrid_cnn_bilstm` 기준(baseline 0.884→0.914)이고 나머지 distill
> 표는 `hybrid_patchtst` 기준이라 절대값 base가 다르다. 비교는 각자 Δbaseline으로.

---

## A. 백본/아키텍처 개선

### A1. 진단 — vanilla PatchTST의 token starvation
- **문제**: window=10인데 non-overlap patch(`patch_len=5, stride=5`)면 토큰 = (10-5)/5+1 = **2개**.
  트랜스포머 self-attention이 무력화됨.
- **증거**: random split 0.97 vs temporal split 0.79 → 진짜 성능은 0.79 (random은 누수 착시).
- 상세: `PATCHTST_NOFREQ_BASELINE_RESULTS.md`

### A2. Overlapping patch (개선의 핵심)
- **변경**: `patch_len=3, stride=1` → 토큰 2개 → **8개(+end pad 9)**, 4배.
- **결과**: no-freq 기준 0.812 → 0.843. FFT와 결합 시 더 큼(A3).
- 코드: `train.py --patch_len 3 --patch_stride 1 --patch_pool mean`, `model.py PatchTSTClassifier`.

### A3. hybrid_patchtst = overlap patch + Frequency branch (teacher 아키텍처)
- **구조(two-branch late fusion)**:
  - time branch: overlap PatchTST → `(B, C·d_model)=256` (patch_num 평균).
  - freq branch(`FFTBranch`): 채널별 rfft → magnitude+phase(2C=8채널) → 1D CNN 2층 → GAP → `(B,128)`.
  - fusion: `concat([256,128])=384` → LayerNorm → Dropout → Linear(11).
- **결과**: hybrid baseline(FFT만) 0.88 → **hpatchfix 0.940** (단일 변경 +0.06).
- **ablation**(`UTICA_PATCH_ABLATION_RESULTS.md`): patchfix 0.80 / crosspatch(채널 attention) 0.875 /
  **hpatchfix 0.940** → hybrid+overlap이 최선.
- 코드: `model.py HybridTimeFreqClassifier`, `FFTBranch`. arch=`hybrid_patchtst`.

### A4. Cross-channel PatchTST (채널 어텐션, 검토됨)
- 각 (channel, patch)를 토큰으로 → attention이 채널 직접 혼합. token=raw+diff+mean+std.
- **결과**: crosspatch base 0.875 < hpatchfix 0.940. 채택 안 함.
- 코드: `model.py CrossChannelPatchTSTClassifier`, arch=`cross_patchtst`.

### A5. Coarse-to-fine patch distillation (검토됨)
- coarse(patch3) teacher → fine(patch2) student로 patch-size 간 distill.
- **결과**: `fine2_base` 0.9486, c2f_t3_s2 0.9448 — sd_init(0.9577) 못 넘음. 채택 안 함.
- 상세: `COARSE_TO_FINE_PATCH_RESULTS.md`

### A6. Frequency branch = PatchTST (★최근 실험, negative)
- **동기**: "time도 freq도 PatchTST로 통일하면?" 검증.
- **구현**: `FFTBranch(arch="patchtst")` — magnitude만(C=4채널), freq축(W//2+1=6)을
  시퀀스로 overlap patch(patch_len=2,stride=1) PatchTST → projection. `train.py --freq_arch patchtst`.
- **결과(3-seed, time=hpatchfix 고정)**: cnn f1 0.9255 vs **patchtst f1 0.8984** (Δ −0.027, 일관 하락).
  weak label_7만 +0.088로 좋고 나머지 손실.
- **원인**: freq축 6 bin → token starvation이 time보다 더 심함. CNN이 짧은 freq에 적합.
- **결론**: freq branch는 CNN 유지. 상세: `FREQ_PATCHTST_RESULTS.md`

### A7. 비교 베이스라인 아키텍처
- ResNet1D 0.9145, TimesNet 0.8897, CNN-BiLSTM 0.817 (모두 100ep).
- 상세: `BASELINE_ARCH_100E_RESULTS.md`, `TIMESNET_RESULTS.md`.

---

## B. 합성 데이터 생성 + supervision policy

### B0. 생성기 — 조건부 1D diffusion (DDPM)
- 클래스 조건부로 윈도우 생성. 클래스당 1000개 → `(11000,10,4)`.
- teacher 합성데이터 정확도 0.834, synth-only 학습만으로 acc 0.875 → **생성기 자체가 충분히 좋음**.
- 코드: `diffusion.py`, `train_diffusion.py`. 산출물 `synthetic_hpatchfix_teacher_1000pc.npz`.

### B1. Hard-label augmentation (무차별) — negative
- 합성을 generator 조건 클래스(hard label)로 그냥 섞음.
- **결과**: hpatchfix 0.940 → **hardaug 0.923** (하락). 합성을 많이 넣는 것 자체는 손해.
- **원인**: 합성 hard label이 teacher decision과 불일치(예: label_8 teacher agreement 0.26,
  label_10 0.41). "합성 라벨은 ground truth가 아니라 uncertain supervision".
- 코드: `train_with_synth.py --mix_mode aug` (hard label).

### B2. Teacher-soft self-distillation (핵심 — 작동)
- real 샘플 = hard CE, **synth 샘플 = frozen teacher soft-label KL**(T=2). hard label 미사용.
- 수식:
  ```
  L_real  = CE(student(x_real), y_real)
  p_t = softmax(teacher(x_syn)/T),  p_s = softmax(student(x_syn)/T)
  L_synth = T^2 · KL(p_t || p_s)
  ```
- 두 변형:
  - `sd_scratch`(랜덤 init): **0.9475** (baseline +0.009, fair improvement).
  - `sd_init`(teacher 가중치 init): **0.9577** (teacher와 동급, "보존"이지 improvement claim 약함).
- weak class 개선: label_7 0.750→0.857, label_9 0.781→0.818, label_10 0.838→0.857.
- 상세: `HPATCHFIX_TEACHER_SD_RESULTS.md`. 코드: `train.py` KD 경로 / `train_with_synth.py`.

### B3. Confusion-aware targeted distillation (확장)
- 약한 클래스(val recall 낮음)에 생성량↑ + KL weight↑.
  ```
  weakness_c = max(0, 1 - recall_c)
  n_c = base * (1 + beta·weakness_c);  w_c = 1 + alpha·weakness_c
  ```
- **결과**: ca_init 0.955 ≈ sd_init 0.958 (전체 평균 개선 없음, 클래스별 거동만 이동).
  label_9/10↑, label_7↓. 단순 teacher-soft로 충분.
- 상세: `CONFAWARE_SD_RESULTS.md`, `make_confusion_aware_plan.py`.

### B4. Synthetic sample selection / filtering — negative
- diffusion 합성을 teacher로 점수화해 3축 필터: class-consistency / realism(kNN) / informativeness(margin).
- **결과**: raw 전체 0.9028 vs sel 0.889~0.875 (모든 정책·2 arch에서 raw ≤ 못 이김).
- **원인**: 생성기가 이미 좋아 버릴 샘플이 적음. 소량 데이터에선 **fidelity보다 diversity가 병목**.
  35~50% 버리면 커버리지 손실 > outlier 제거 이득.
- **언제 유효**: 생성기가 약/noisy하거나 합성:실제 비율을 크게 키울 때. 현재는 보류.
- 상세: `SELECTION_NOTES.md`, `select_synthetic.py`.

### B5. Closed-loop targeted augmentation (★가장 강하고 일관된 gain)
- **아이디어**: student/teacher를 val로 진단 → 약한 클래스에 diffusion 생성 예산 집중(student-aware,
  on-policy 정신). hard label, **filter 끔, full budget(~8x)**.
- **결과(hybrid_cnn_bilstm, 3-seed)**:
  - baseline 0.8843, uniform-NF 0.8889(+0.005), **targeted-NF 0.9144(+0.030)**.
  - targeted − uniform = +0.0255 (3 seed 전부 양수).
- **두 함정(초기 실패 원인)**: (1) filter가 gain을 죽임(B4와 일치), (2) 이 데이터는 합성 많을수록 좋음
  (~8x > ~1.4x). 둘 제거하고 targeting만 분리하니 모든 seed win.
- **명칭(write-up)**: `student-aware (on-policy) targeted augmentation`. distill/OPD 정신이
  분류기에서 실제 작동한 첫 케이스.
- 상세: `CLOSED_LOOP_DESIGN.md`, `CLOSED_LOOP_RESULTS.md`, `closed_loop.py`.
  (TODO: `closed_loop.py`에 `--no_filter` 플래그 추가, R=2,3 반복 확장.)

---

## C. Distillation 종류 (teacher 종류별 분류)

분류 기준 = **teacher가 무엇인가** + **student가 보는 입력이 무엇인가**.

### C1. SDFT (self-distillation fine-tuning) — 보류
- teacher = 자기 자신 또는 student의 **EMA**. student = noisy 입력. real-data 합성 불필요.
- 수식: `L = α·CE(student(x_noisy), y) + (1-α)·warmup·T²·KL(p_T(x_clean) || p_S(x_noisy))`.
- **개선한 점(코드 가치)**: naive from-scratch SDFT는 confirmation-bias collapse(최악 0.333).
  EMA teacher + eval-mode target + KL warmup + `--init_from`(수렴 baseline에서 FT)로 **collapse 해결**
  (aggressive a0.5 T4: 0.660 → 0.736).
- **성능 결론**: 다중 seed 평균은 baseline 못 넘음 (mean Δ ≈ −0.012). baseline 약할 때만 도움 =
  regression-to-mean. **self-distillation은 이 데이터의 성능 레버 아님.**
- 코드: `train.py train_epoch_sdft`, `--sdft --sdft_ema 0.999 --sdft_kl_warmup`. 상세: `SDFT_NOTES.md`.

### C2. Online SDFT (EMA-teacher clean→noisy consistency) — preliminary
- C1의 online 버전. teacher = student EMA(매 스텝 갱신), clean→teacher / noisy→student consistency.
- **분류 주의**: teacher가 student-EMA이므로 **OPD가 아니라 Mean-Teacher식 self-distillation**.
- **결과(hpatchfix, init_from base, 30ep)**:
  - base 0.9377, ce_noise_ft 0.9567, **light_a09_t2_p03 0.9619**(best), strong 0.9472(과함).
  - light가 noise-only 대비 +0.0052 (작지만 일관). **gain 대부분은 noise augmentation 효과**.
- 권장 기본: `--noise --noise_mode real --noise_p 0.3 --sdft --sdft_alpha 0.9
  --sdft_temperature 2.0 --sdft_ema 0.999 --sdft_kl_warmup 5 --lr 3e-4 --epochs 30 --init_from <base>`.
- 상세: `ONLINE_SDFT_EXPERIMENT_DESIGN.md`, `ONLINE_SDFT_RESULTS.md`.

### C3. GKD / OPD 일반형 (forward/reverse/JSD KD) — negative
- **독립 frozen teacher**가 student가 보는 augmented 입력을 채점 = 진짜 OPD.
- 일반형 손실(분류 축약):
  ```
  L = α·CE(f_S(x_aug), y) + (1-α)·[ (1-λ)·Div(x_clean) + λ·Div(x_aug) ]
  Div = JSD_β = T²·[ β·KL(p_T||M) + (1-β)·KL(p_S||M) ],  M = β·p_T + (1-β)·p_S
  β=1 forward-KL(mode-covering, 기존 KD) / β=0 reverse-KL(mode-seeking) / β=0.5 symmetric
  λ = on-policy 비율(λ=1 순수 aug). gradient는 student로만, p_T는 no_grad. T² 스케일.
  ```
- **결과**: 모든 변형(forward/reverse/JSD)이 baseline 이하. **reverse-KL이 최악**
  (mode-seeking → diversity 손실, 이 데이터는 diversity가 병목).
- 코드: `train.py gkd_divergence`, `train_epoch_kd`, `--gkd_beta --gkd_lambda --kd_teacher`.
  상세: `OPD_SDFT_DESIGN.md` (+ 결과는 SDFT/CLAUDE.md 교훈에 통합).

---

## D. 정규화/증강 (입력 노이즈)

### D1. 물리 기반 노이즈 augmentation (label-preserving)
- diffusion 생성과 별개. **원본 real 윈도우에 학습 시점 노이즈를 더해 변형**(라벨 유지). val/test 비활성.
- 3단계 강도(`noise_augment.py`):
  - **TimeSeriesNoise** (기본 4종): Gaussian(thermal), amplitude jitter(±5%, 배터리/온도),
    DC bias(calibration drift), spike(PWM/EMI).
  - **RealEnvNoise** (종합): +drift(저주파 sin), pink(1/f), hum(60Hz), boundary bias(클래스 경계 침범),
    masking(센서 dropout).
  - **DomainRandomNoise** (domain randomization): 윈도우마다 노이즈 환경 랜덤 — AR(1) 자기상관,
    채널간 상관(케이블/EMI), SNR 랜덤화(×0.5~3), LP smoothing, saturation, quantization(ADC bit).
- **역할**: 소량 데이터에서 측정 노이즈 강건성↑. online SDFT(C2)의 gain 상당 부분이 이 효과.
- 코드: `train.py --noise --noise_mode {time,real,domain} --noise_p`.

---

## E. Foundation 비교 (Utica)

- frozen time-series foundation(Utica) + linear/MLP probe.
- **결과**: utica concat_mlp 0.9577 ≈ 우리 sd_init 0.9577 → **별도 대규모 사전학습 없이 동급**.
- 메시지: 작은 도메인 데이터에선 도메인 맞춤 개선이 거대 foundation과 맞먹는다.
- 상세: `UTICA_PROBE_RESULTS.md`, `train_utica_probe.py`.

---

## F. 학습 루프 최적화 (인프라)
- 시드 고정(`--seed` → python/numpy/torch), grad clip, cosine LR, EMA buffer 동기화 등.
- 상세: `REPORT.md`.

---

## 핵심 교훈 (정직한 요약)

1. **backbone이 먼저다**: 약한 backbone(hybrid_cnn_bilstm) 위에선 모든 distillation이 baseline 이하.
   hpatchfix로 고친 뒤에야 teacher-soft KD가 작동. "distillation을 쓸지보다 작동할 backbone을 먼저."
2. **합성은 양보다 policy**: 무차별 hardaug는 손해, teacher-soft / 표적(closed-loop)만 이득.
   "synthetic은 uncertain evidence로 다뤄라."
3. **이 데이터는 diversity가 병목**: filter/selection은 손해, reverse-KL(mode-seeking)은 최악,
   합성은 많을수록(~8x) 좋음.
4. **self-distillation(SDFT/online SDFT)은 성능 레버가 아님**: collapse는 고쳤으나 다중 seed gain 없음.
   gain처럼 보인 건 noise augmentation 효과 또는 regression-to-mean.
5. **closed-loop targeted augmentation이 가장 강하고 일관**(+0.030, 3 seed 전부): 진단→표적생성→증강.
6. **짧은 축엔 Transformer 부적합**: time축은 overlap patch로 살렸지만 freq축(6 bin)은 CNN이 낫다.
7. **항상 3-seed**: 단일 seed(test 144) 함정을 반복 경험. 모든 판단은 ≥3 seed 평균으로.

---

## 가장 강한 조합 (현재 best 레시피)

- backbone: `hybrid_patchtst`, overlap patch(`patch_len=3,stride=1,pool=mean`), freq branch = **CNN**.
- supervision: teacher-soft self-distillation(`sd_init`, T=2) → 0.958, 또는
  closed-loop targeted augmentation(no-filter, full budget) → 가장 일관된 +0.030.
- (옵션) online SDFT light + 물리 노이즈 → 0.962(preliminary).

## 열린 TODO / 다음 방향
- `closed_loop.py --no_filter` 플래그, R=2/3 반복(재진단 표적), 다른 arch에서 targeting>uniform 재현.
- class별 teacher-agreement 기반 generation ratio/filter threshold 자동조정.
- physics-informed 파생 채널(VBAT·ILOAD, VBAT·ICHG, ICHG−ILOAD).
- public TS benchmark + bootstrap CI/통계 검정(test set 작음 보완).

---

## 개별 문서 인덱스
- 백본: `PATCHTST_NOFREQ_BASELINE_RESULTS.md`, `UTICA_PATCH_ABLATION_RESULTS.md`,
  `COARSE_TO_FINE_PATCH_RESULTS.md`, `FREQ_PATCHTST_RESULTS.md`, `BASELINE_ARCH_100E_RESULTS.md`,
  `TIMESNET_RESULTS.md`
- 합성/증강: `HPATCHFIX_TEACHER_SD_RESULTS.md`, `CONFAWARE_SD_RESULTS.md`, `SELECTION_NOTES.md`,
  `CLOSED_LOOP_DESIGN.md`, `CLOSED_LOOP_RESULTS.md`
- distillation: `SDFT_NOTES.md`, `OPD_SDFT_DESIGN.md`, `ONLINE_SDFT_EXPERIMENT_DESIGN.md`,
  `ONLINE_SDFT_RESULTS.md`
- foundation/인프라: `UTICA_PROBE_RESULTS.md`, `REPORT.md`
- 발표: `PRESENTATION_GUIDE.md`, `presentation/research_status_seminar.md`
