# In-Network Self-Distillation & Looped Architectures for Data-Scarce Short-Window Time-Series Classification

frontier 자기증류(SDFT)·looped/recurrent(LoopViT, Parcae)·knowledge distillation 기법을
**소량(클래스당 1분)·단윈도우(W=10) 4채널 시계열 11-class 분류**에 체계적으로 이식하고,
**언제 작동하고 왜 실패하는지**를 통제된 ablation(3-seed, temporal split)으로 규명한 연구.

## 핵심 메시지 (TL;DR)

소량·단윈도우 시계열에서 in-network/looped 자기증류는 **약한 backbone을 안정화할 뿐
강한 backbone을 넘지 못하며**, 진짜 성능 레버는 (i) overlapping-patch로 token starvation
해소, (ii) diffusion 합성 + 외부 teacher-soft KD다. 자기증류의 효용은 **backbone 강도와
데이터 양에 대한 조건부**임을 통제 실험으로 보인다.

## 결과 한눈 (3-seed 평균, weighted F1)

| 방법 | F1 | 비고 |
|---|---:|---|
| vanilla PatchTST (단윈도우) | 0.81 | token starvation (패치 2개) |
| + overlapping patch + FFT (hpatchfix) | 0.94 | backbone 개선이 핵심 |
| base_L4 (깊이↑, 원본만) | 0.948 | 깊이는 L4서 포화 |
| in-network SDFT (ema_L2) | 0.945 | 약 backbone 안정화 (+0.020), 강 backbone(L4)선 −0.025 |
| looped (gated, 원본만) | 0.55 | ‖h‖ 폭발 아닌 과적합으로 실패 |
| looped + 합성 KD (hpatchfix teacher) | 0.72 | 데이터↑로 회복 but 부족 |
| looped + 합성 KD (self-EMA teacher) | 0.46 | 외부 teacher 없으면 더 나쁨 |
| **일반 PatchTST + 합성 teacher-soft KD** | **0.955** | 정답. foundation(Utica 0.958) 동급 |

## 저장소 구조

```
.
├── README.md                # 이 파일
├── METHODS_MASTER.md        # 적용한 모든 방법 종합 레퍼런스
├── PAPER_OUTLINE.md         # empirical study 논문 outline (8~10p)
├── src/                     # 핵심 코드
│   ├── model.py             # PatchTST / hpatchfix / LoopedPatchTST / FFTBranch
│   ├── train.py             # 학습 루프 (in-network SDFT, GKD/OPD, multi-exit)
│   ├── train_with_synth.py  # 합성 teacher-soft KD (frozen / EMA teacher relabel)
│   ├── dataset.py, metrics.py, noise_augment.py, diffusion.py
├── scripts/                 # 실행·집계
│   ├── run_dual_gpu.sh      # 2-GPU 분할 러너 (jobfile 라운드로빈)
│   ├── gen_*_jobs.py        # ablation jobfile 생성기
│   ├── collect_*.py         # 결과 집계 (3-seed 평균/약한클래스/Δ)
│   ├── probe_loop_norm.py   # per-step ‖h_t‖ 폭발 관찰
│   └── eval_dynamic_exit.py # Phase 2 Dynamic Exit 평가
└── docs/                    # 단계별 설계·결과 문서
    ├── LOOPED_SDFT_DESIGN.md            # 통합 설계 (Phase 0/1/2)
    ├── LOOPED_SDFT_PHASE0_RESULTS.md    # multi-exit SD
    ├── LOOPED_SDFT_PHASE0_ABLATION.md   # SDFT ablation (EMA/aux/λ/L)
    ├── LOOPED_SDFT_PHASE12_RESULTS.md   # looped backbone + Dynamic Exit
    ├── LOOPED_NORM_PROBE.md             # ‖h‖ 폭발 진단 (gated 재설계)
    ├── LOOPED_GATED_RESULTS.md          # gated 학습 (폭발 제거 반증)
    └── LOOPED_SYNTH_KD_RESULTS.md       # 합성 KD (frozen/EMA, epoch 스윕)
```

## 핵심 발견

1. **Token starvation**: 단윈도우(W=10) non-overlap patch는 토큰 2개 → attention 무력화.
   overlapping patch(p3/s1)로 8개 확보 → 0.81→0.94.
2. **In-network SDFT는 조건부**: EMA teacher가 약한 backbone(L2)은 +0.020 안정화하지만
   강한 backbone(L4)에선 −0.025. 강한 KL은 collapse.
3. **Looped 실패 원인 분리**: per-step ‖h_t‖ probe로 update/prenorm은 폭발(8~12x),
   gated convex update는 안정(1.4x). 그러나 gated로 폭발을 없애도 성능 회복 안 됨
   → **병인은 ‖h‖ 폭발이 아니라 소량 데이터 과적합**.
4. **합성 KD가 유일한 일관 레버**: 일반 backbone + diffusion 합성 teacher-soft KD = 0.955
   (foundation 동급). looped는 합성 KD로 회복되나(과적합 진단 확증) 일반 backbone 못 넘음.

## 재현 (요약)

데이터는 본 레포에 포함하지 않음(내부 센서 CSV). 학습은 `src/`를, ablation은 `scripts/`를
참조. 아래 명령은 원 작업 디렉토리(`src/`의 스크립트와 데이터·산출물이 같은 폴더에 있는
플랫 레이아웃)를 가정하므로, 이 레포 레이아웃에서 돌리려면 경로를 맞춰야 한다. 2-GPU 병렬:

```bash
PYSCRIPT=train.py            bash scripts/run_dual_gpu.sh scripts/jobs_looped_ablation.txt
PYSCRIPT=train_with_synth.py bash scripts/run_dual_gpu.sh scripts/jobs_synloop.txt
python scripts/collect_synloop_results.py
```

평가는 항상 3-seed(42/1/7) 평균, temporal split, val/test는 real만(합성 미포함).

## 레퍼런스
- PatchTST (arXiv 2211.14730)
- LoopViT (github.com/WenjieShu/LoopViT)
- Parcae: Scaling Laws for Stable Looped Language Models (arXiv 2604.12946)
- SDFT (sail-sg), Mean-Teacher / BYOT, GKD (arXiv 2306.13649)
