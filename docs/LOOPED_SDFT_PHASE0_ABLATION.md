# Phase 0 재시도 — multi-exit SD ablation 결과

작성 2026-06-29. `LOOPED_SDFT_PHASE0_RESULTS.md`의 무효과를 깨려고 4축을 ablation.
모두 hybrid_patchtst, freq=CNN, overlap patch(p3/s1/mean), 50ep, 3-seed(42/1/7).
2-GPU 병렬 실행(`scripts/run_dual_gpu.sh`).

## ablation 축
- 가설 a (deep/shallow 표현차 작음) → e_layers=4 (sd_L4)
- λ 스윕 → lam2_L2, lam4_L2
- 가설 b (teacher 약함) → EMA teacher (ema_L2)
- 직접 supervision → 중간 exit aux CE (auxce_L2)
- 대조군: base_L2(=직전 mexit_base), base_L4

## 결과 (3-seed 평균 ± std)

| arm | L | acc | F1 | Δf1 vs 동일-L base |
|---|---:|---:|---:|---:|
| base_L2 | 2 | 0.9282 ± 0.0087 | 0.9255 ± 0.0091 | — |
| **base_L4** | 4 | **0.9491 ± 0.0033** | **0.9479 ± 0.0029** | — |
| base_L6 | 6 | 0.9468 ± 0.0087 | 0.9454 ± 0.0088 | — (L4서 포화) |
| sd_L4 | 4 | 0.8403 ± 0.0247 | 0.8401 ± 0.0244 | **−0.108 (붕괴)** |
| lam2_L2 | 2 | 0.9005 ± 0.0118 | 0.8958 ± 0.0108 | −0.030 |
| lam4_L2 | 2 | 0.8009 ± 0.0143 | 0.7926 ± 0.0187 | **−0.133 (붕괴)** |
| ema_L2 | 2 | 0.9468 ± 0.0065 | 0.9454 ± 0.0073 | +0.020 |
| **ema_L4** | 4 | 0.9259 ± 0.0215 | 0.9233 ± 0.0237 | **−0.025 (강한 backbone선 손해)** |
| auxce_L2 | 2 | 0.9329 ± 0.0118 | 0.9306 ± 0.0133 | +0.005 |

## 약한 클래스 F1 (3-seed 평균)

| arm | label_7 | label_8 | label_9 | label_10 |
|---|---:|---:|---:|---:|
| base_L2 | 0.7123 | 0.8500 | 0.7455 | 0.8265 |
| base_L4 | 0.8554 | 0.8799 | 0.7850 | 0.8549 |
| ema_L2 | 0.8178 | 0.9209 | 0.7876 | 0.8190 |
| sd_L4 | 0.7191 | 0.7691 | 0.6195 | 0.5770 |
| lam4_L2 | 0.5741 | 0.7630 | 0.5714 | 0.6036 |

## 핵심 발견

1. **deep→shallow KL을 세게 걸면 collapse한다.** lam4_L2(−0.133), sd_L4(−0.108) 모두 붕괴.
   λ↑ 또는 exit↑로 KL을 강화할수록 악화 → 가설 a(표현차 벌리기)는 **틀렸고 오히려 해롭다**.
   최종 exit teacher가 얕은 exit을 강하게 끌면 backbone 전체가 teacher의 미성숙 예측에
   끌려가 무너지는 confirmation-bias collapse(naive SDFT와 동일 병리).

2. **EMA teacher가 무효과를 깼다 (가설 b 적중).** ema_L2 f1 0.9454, base_L2 대비 +0.020,
   std도 감소(0.0091→0.0073). 약한 클래스 전반 개선(label_7 +0.106, label_8 +0.071).
   stop-grad 최종 exit보다 EMA가 안정적 타깃이라는 online SDFT 교훈과 일치.

3. **진짜 레버는 단순 깊이 증가다.** base_L4(0.9479)가 모든 SD arm보다 높다. L2→L4 +0.022.
   base_L6(0.9454)에서 포화 → 깊이 효과는 L4가 천장(소량 데이터 한계).

4. **EMA gain은 "안정화"이지 진짜 distillation 이득이 아니다 (공정 비교로 확정).**
   - L2: base_L2 0.9255 → ema_L2 0.9454 (+0.020). 얕고 불안정한 모델을 안정화.
   - **L4: base_L4 0.9479 → ema_L4 0.9233 (−0.025).** 이미 안정적인 강한 backbone엔 손해.
     std도 0.0029→0.0237로 폭증(불안정 유발).
   → EMA teacher는 **약한 모델을 깊은 모델 수준으로 끌어올릴 뿐**, 강한 backbone을 못 넘는다.

## 최종 결론

Phase 0의 **모든 변형(stop-grad / EMA / aux CE / λ 스윕 / L4)이 동일 패턴**을 따른다 —
self-distillation은 약한 모델을 안정화할 뿐 강한 backbone(base_L4)에선 손해이고, KL을 세게
걸면 collapse한다. 이는 프로젝트 누적 교훈(`SDFT_NOTES.md`, `ONLINE_SDFT_RESULTS.md`,
`LOOPED_SDFT_PHASE0_RESULTS.md`)의 **세 번째 재확인**이다.

→ 설계 §8 기준, **multi-exit/in-network SDFT 경로는 비유망으로 종결.** "SDFT를 먼저 얹는"
looped 통합은 이 데이터에서 작동하지 않는다. 만약 looped를 계속 본다면
**SDFT 없이 순수 looped backbone(weight-tied, Parcae 안정화)이 base_L4(0.948)를 넘는지**가
유일하게 남은 검증 포인트다(단, 깊이가 L4서 포화하므로 weight-tied 반복도 큰 gain은 기대난망).

## Artifacts
- 스크립트 `scripts/run_multiexit_ablation_gpu1.sh`, `scripts/run_dual_gpu.sh`,
  jobfile `scripts/jobs_mxab_*.txt`
- 집계 `collect_mxab_results.py`
- 메트릭 `results_json/test_metrics_mxab_*_e50_s*.json`, base_L2=`..._mexit_base_*`
