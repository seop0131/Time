# Looped Architecture × SDFT — 설계 문서 (탐색/타당성 검증)

작성 2026-06-29. 목표 = LoopViT / Parcae식 **looped architecture**를 우리 **SDFT 학습방법**과
통합해, 이 데이터(window=10, 소량, test 144)에서 의미가 있는지 단계적으로 검증.
선행 결론은 누적 교훈(`METHODS_MASTER.md`)을 안전장치로 강제하며 진행.

---

## 1. 두 레퍼런스 정리

### LoopViT (github.com/WenjieShu/LoopViT)
- weight-tied **Hybrid Block**(conv + global attention)을 forward에서 T번 재귀.
- depth를 파라미터와 분리("scaling time over space").
- **Dynamic Exit("predictive crystallization")**: loop가 진행되며 entropy 낮아지면(예측 안정화)
  조기 종료. parameter-free.
- 성과: 18M로 73M ensemble 능가 (ARC-AGI).

### Parcae: Scaling Laws for Stable Looped LMs (arXiv 2604.12946, UCSD/Together)
- weight-tied Transformer layer를 T번 재귀 = 같은 looped architecture.
- **핵심 기여 = 안정화**: looping을 residual stream 위의 nonlinear time-variant 동역학계로 보고,
  불안정(residual explosion / loss spike)의 원인을 **injection parameter의 큰 spectral norm**으로 규명.
  → **negative diagonal parameterization의 discretization으로 injection spectral norm 제약**.
- **scaling law**: optimal recurrence ∝ FLOPs^0.40, optimal tokens ∝ FLOPs^0.78
  → **looping과 데이터를 함께 늘려라**. 770M로 1.3B급.

### 공통 / 우리에게 주는 것
- 둘 다 looped **architecture** (forward 안 T번 반복). 둘 다 distillation은 다루지 않음.
- LoopViT → step별 출력 + 조기종료 메커니즘.
- Parcae → **소량/짧은 시퀀스에서 loop 발산을 막는 안정화 레시피**(우리에게 가장 중요).

---

## 2. looped architecture loop vs SDFT loop — 직교성 (통합의 근거)

| 축 | looped architecture (LoopViT/Parcae) | SDFT / closed-loop |
|---|---|---|
| loop 단위 | **공간적** — forward 안에서 latent를 T번 정제 | **시간적** — 학습 라운드 R번 반복 |
| 대상 | 한 입력의 inference compute | 학습 절차(teacher 갱신/relabel) |
| 코드 위치 | `model.py` backbone | `train.py` 학습 루프 |

→ 둘은 **직교**. 자연스러운 통합 = looped backbone의 **중간 loop 출력들 사이에 self-distillation**을
건다. 즉 loop를 깊이용으로만 쓰지 않고 **deep loop step(teacher) → shallow loop step(student)**
consistency를 SDFT loss로 강제. 세 레퍼런스가 한 손실에 모임.

---

## 3. 통합 아키텍처 — LoopedPatchTST + in-network SDFT

### 3.1 Looped backbone
overlap patch 임베딩 후 weight-tied 블록 f_θ를 T번 적용. 매 step 입력 재주입(injection).
```
e        = PatchEmbed(x)                 # (B, C, n_patch, d_model), overlap patch 유지
h_0      = e
for t in 0..T-1:
    h_{t+1} = h_t + g( f_θ(h_t) + W_inj · e )   # residual + injection
    z_t     = head(pool(h_{t+1}))               # step별 분류 logit
```
- f_θ = 기존 TSTiEncoder 블록 1개를 weight-tie (e_layers를 loop로 대체).
- **injection W_inj** = Parcae 안정화 대상. negative-diagonal/spectral-norm 제약.
- head = 기존 fusion head 공유(또는 step별 LayerNorm + 공유 Linear).
- pool = patch_num mean (기존과 동일).

### 3.2 In-network SDFT 손실 (deep→shallow self-distill)
```
L = CE(z_T, y)                                            # 최종 step hard label (anchor)
  + λ · Σ_{t<T}  warmup · τ² · KL( sg[p_τ(z_T)] ‖ p_τ(z_t) )   # deep teacher → shallow student
```
- p_τ = softmax(·/τ). sg = stop-gradient (또는 EMA copy of z_T) → teacher 신호 안정화.
- 이렇게 하면 SDFT의 clean/noisy consistency 대신 **deep/shallow consistency**가 됨.
- **변형 가능**: teacher 신호를 (a) 최종 step z_T sg, (b) z_t들의 EMA, (c) 별도 EMA-weight backbone.
  C2(online SDFT)의 EMA-teacher 교훈 재사용.

### 3.3 Dynamic Exit (LoopViT) — 선택
- inference 시 entropy(z_t) < 임계면 조기 종료. parameter-free.
- 학습엔 영향 없음(분석/효율용). 처음엔 끄고 T 고정으로 검증.

---

## 4. 안정화 전략 (Parcae 차용 + 우리 교훈)

소량 데이터에서 loop는 발산/과적합 위험. 다음을 기본 ON.
1. **Injection spectral-norm 제약** (Parcae): W_inj를 negative-diagonal 파라미터화 또는
   `nn.utils.parametrizations.spectral_norm`. residual explosion 방지.
2. **작은 T(2~3)부터**: METHODS_MASTER 교훈(소량+큰 용량=손해). Parcae scaling law도
   "loop·데이터 동반 증가"인데 우리는 데이터가 적으니 loop도 작게.
3. **KL warmup**: 초기 N epoch λ 0→1 ramp (SDFT collapse 방지, `--sdft_kl_warmup`과 동일 패턴).
4. **grad clip 1.0**, cosine LR — 기존 그대로.
5. **init_from**: 수렴된 hpatchfix base에서 fine-tune 시작도 한 변형으로 시험.

---

## 5. 단계적 실험안 (모두 hybrid_patchtst, freq=CNN 고정, 3-seed 42/1/7, merge11, temporal)

### Phase 0 — multi-exit SD (위험 0, 코드 최소) ※ looped backbone 없이
기존 hpatchfix는 e_layers=2. 이를 exit 2개로 보고 layer1→layer2 KL만 추가.
looped 핵심 효과(중간표현 deep→shallow self-distill)의 *방향성*을 0 비용 확인.
| arm | 설명 |
|---|---|
| hpatchfix base | 대조군 (0.940/0.938) |
| multi_exit_sd | + Σ KL(sg p(z_L) ‖ p(z_{L-1})) |
- 성공 기준 = 3-seed 평균이 base 이상(최소 비열세). negative면 looped 전면 구현 보류 근거.

### Phase 1 — Looped backbone (Phase 0 긍정 시)
| arm | T | injection 안정화 | SDFT |
|---|---:|---|---|
| loop_T2_plain | 2 | off | off (순수 looped, ablation) |
| loop_T2_stab | 2 | on (Parcae) | off |
| loop_T2_sd | 2 | on | on (deep→shallow KL) |
| loop_T3_sd | 3 | on | on |
- 비교 대상 = hpatchfix base(0.938), sd_init(0.958), closed-loop(+0.030).
- ablation 축: T(2/3), 안정화(on/off), SDFT(on/off) → 무엇이 효과인지 분리.

### Phase 2 — (선택) Dynamic Exit / closed-loop(R≥2) 결합
- Phase 1이 살아나면 LoopViT 조기종료(효율) + 시간축 closed-loop(R=2,3) 결합.

---

## 6. 구현 위치 (코드)
- `model.py`: `LoopedPatchTSTClassifier` 신설. PatchEmbed는 기존 PatchTSTClassifier에서 분리,
  TSTiEncoder 블록 1개를 weight-tie해 T번 호출. injection W_inj + spectral-norm.
  step별 logit 리스트 반환(`forward(x, return_all_steps=True)`).
- `HybridTimeFreqClassifier`: time_arch="looped_patchtst" 분기 추가(중간 step feature는 최종 step 사용,
  in-network SDFT는 time branch 내부에서 처리하거나 hybrid 우회).
- `train.py`: `train_epoch_looped_sd` 추가 — step logits로 CE(z_T)+Σ KL. args
  `--loop_T`, `--loop_inject_stab`, `--loop_sd_lambda`, `--loop_sd_temp`, `--loop_kl_warmup`.
- 원본 backbone(`PatchTST/PatchTST_supervised/layers/PatchTST_backbone.py`)은 수정 금지,
  TSTiEncoder를 우리 쪽에서 재귀 호출하는 wrapper로 감쌈.

---

## 7. 위험 / 예상 (정직하게)
- **SDFT는 이 데이터의 성능 레버가 아니었음**(`SDFT_NOTES.md`): deep→shallow KL도 gain이 작을 수 있음.
- **소량+용량↑ 손해 패턴**(freq-PatchTST −0.027, SDFT collapse): T를 키우면 악화 가능 → T 작게.
- 그럼에도 **looped+안정화는 새 축**이라 negative여도 분석 가치(목표 2 = 방법론 기여).
  특히 **Parcae 안정화 ↔ SDFT collapse**가 만나는 지점(injection spectral norm이 self-distill
  안정성에 주는 효과)이 학술적으로 흥미.
- 반드시 3-seed. T·안정화·SDFT를 ablation으로 분리해 "무엇이 효과/해악인지" 규명.

---

## 8. 성공/중단 기준
- Phase 0 multi_exit_sd ≥ base (3-seed) → Phase 1 진행. 아니면 looped 보류, 설계 노트만 남김.
- Phase 1에서 loop_T2_sd > base이고 안정화 on>off면 통합 유효. sd_init(0.958) 초과면 강한 결과.
- 어느 단계든 단일 seed 반전은 무시, 3-seed 일관성만 신뢰.

## 9. 레퍼런스
- LoopViT: https://github.com/WenjieShu/LoopViT
- Parcae: https://arxiv.org/abs/2604.12946 , https://github.com/sandyresearch/parcae
- 우리 자산: `METHODS_MASTER.md`, `SDFT_NOTES.md`, `ONLINE_SDFT_RESULTS.md`, `CLOSED_LOOP_RESULTS.md`
