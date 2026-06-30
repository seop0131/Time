# Looped × SDFT — Phase 1/2 결과 + 전체(Phase 0/1/2) 비교

작성 2026-06-29. 설계 `LOOPED_SDFT_DESIGN.md` Phase 1(looped backbone)·Phase 2(Dynamic Exit).
모두 3-seed(42/1/7), 50ep, merge11, temporal split. 2-GPU 병렬(`run_dual_gpu.sh`).

## 구현 (model.py / train.py)

- **`LoopedPatchTSTClassifier`** (Phase 1): overlap patch 임베딩 후 **하나의** TSTEncoderLayer
  f_θ를 T번 재귀. 매 step 임베딩 e를 injection 재주입.
  `h_{t+1} = h_t + LN( f_θ(h_t) + W_inj·e )`. weight-tied → **params가 T와 무관(41,868 고정)**.
- **Parcae injection 안정화** (`--loop_inject_stab`): W_inj를 negative-diagonal init +
  `spectral_norm`으로 큰 spectral norm 억제(residual explosion 방지).
- **loop-step SDFT** (`--multiexit_sd`): 최종 step → 얕은 step deep→shallow KL.
- **Dynamic Exit** (Phase 2, `forward_dynamic_exit`): entropy<thr면 조기종료. 추론 전용.
- arch `looped_patchtst`, args `--loop_T`, `--loop_inject_stab`.

## Phase 1 결과 (3-seed 평균 ± std)

| arm | T | params | acc | F1 | Δf1 vs base_L4 |
|---|---:|---:|---:|---:|---:|
| base_L2 (Phase 0, 일반 PatchTST) | — | ~103K | 0.9282 ± 0.0087 | 0.9255 | −0.022 |
| **base_L4 (Phase 0, 일반 PatchTST)** | — | ~170K | **0.9491 ± 0.0033** | **0.9479** | (기준) |
| loop_T2_plain | 2 | 41,868 | 0.7292 ± 0.0150 | 0.7056 | −0.242 |
| loop_T3_plain | 3 | 41,868 | 0.6829 ± 0.0545 | 0.6658 | −0.282 |
| loop_T2_stab | 2 | 41,868 | 0.7708 ± 0.0260 | 0.7481 | −0.200 |
| loop_T3_stab | 3 | 41,868 | 0.6065 ± 0.0489 | 0.5722 | −0.376 |
| loop_T6_stab | 6 | 41,868 | 0.3843 ± 0.0853 | 0.3208 | −0.627 |
| loop_T3_stab_sd | 3 | 41,868 | 0.6759 ± 0.0606 | 0.6449 | −0.303 |

## 핵심 발견

1. **looped backbone은 이 데이터에서 완전히 실패.** 최고(loop_T2_stab 0.748)도 base_L4(0.948)
   대비 −0.20, 일반 base_L2(0.926)에도 한참 못 미친다.

2. **T가 커질수록 급격히 붕괴.** T2(0.748) → T3(0.572) → T6(0.321). weight-tied 반복이
   깊어질수록 악화. std도 큼(0.03~0.085) → 불안정.

3. **원인 = 과적합(발산 아님).** 학습 곡선(T6_stab): train acc 0.70→0.83 상승하는데
   val acc는 ep10 0.77 → ep50 0.50~0.55로 **하락**(val loss 0.85→1.2 상승).
   weight-tied로 파라미터(41K)는 안 늘어도 **effective depth가 표현력을 키워** 소량
   데이터(train 1320)에 과적합. METHODS_MASTER "소량+큰 용량=손해"의 가장 극적 사례.

4. **Parcae 안정화는 부분적.** T2에서만 도움(plain 0.706 → stab 0.748). T3/T6에선
   과적합을 못 막음(spectral-norm은 발산은 억제하나 과적합은 별개 문제).

5. **loop-step SDFT도 못 살림.** T3_stab_sd(0.645)가 T3_stab(0.572)보다 약간 나을 뿐
   여전히 붕괴. Phase 0 결론(SDFT는 강한 backbone에서 무익)과 일치.

## Phase 2 — Dynamic Exit

메커니즘은 동작(`eval_dynamic_exit.py`). T6_stab s7 측정:

| entropy_thr | mean_steps | accuracy |
|---:|---:|---:|
| 0.0 (끝까지) | 6.00 | 0.487 |
| 0.1 ~ 1.0 | 6.00 | 0.487 |
| 99 (항상 첫 step) | 1.00 | 0.408 |

조기종료가 thr 0.1~1.0에서 **거의 안 걸린다(mean_steps 6.0 고정)** — 과적합된 looped 모델의
예측 entropy가 높아(불확실) "predictive crystallization"이 일어나지 않기 때문. 즉 LoopViT의
조기종료는 모델이 확신을 가질 때 의미가 있는데, 이 데이터의 looped 모델은 그 전제를 못 채운다.
Phase 1 실패와 일관. (looped 자체가 base에 못 미쳐 실용 의미는 약함.)

## 전체 비교 (Phase 0/1/2)

| 접근 | 최고 F1 | vs base_L4(0.948) | 결론 |
|---|---:|---|---|
| 일반 PatchTST 깊이↑ (base_L4) | **0.948** | 기준 | **진짜 레버는 단순 깊이** |
| Phase 0: multi-exit SD (ema_L2) | 0.945 | −0.003 | 약한 모델 안정화일 뿐, base_L4 못 넘음 |
| Phase 1: looped backbone (T2_stab) | 0.748 | −0.200 | 과적합으로 완전 실패 |
| Phase 1+SDFT (T3_stab_sd) | 0.645 | −0.303 | 실패 |

## 최종 결론

**looped architecture(LoopViT/Parcae식) × SDFT 통합은 이 데이터(window=10, train 1320,
test 144)에서 작동하지 않는다.** 세 Phase 모두 동일 근본 원인 —
- self-distillation은 약한 모델 안정화일 뿐 강한 backbone을 못 넘고(Phase 0),
- looped 깊이 확장은 소량 데이터에서 과적합으로 붕괴한다(Phase 1, T↑일수록 심함).

Parcae scaling law("loop와 데이터를 함께 늘려라")가 정확히 우리 발목을 잡았다 — 데이터가
극소량이라 loop를 늘릴 여력이 없다. Parcae가 770M/대규모 토큰에서 성립한 것과 정반대 레짐.

**기여 서사(정직한 negative)**: "frontier looped LM 기법을 소량·단윈도우 시계열에 이식 →
weight-tied depth가 과적합을 유발, 깊이는 일반 L4에서 이미 포화. 이 레짐에선 도메인 맞춤
backbone(hpatchfix) + teacher-soft/closed-loop 증강이 옳고, looped/SDFT는 부적합." 이는
프로젝트 누적 교훈의 네 번째·다섯 번째 재확인.

## Artifacts

- 설계 `LOOPED_SDFT_DESIGN.md`, Phase 0 `LOOPED_SDFT_PHASE0_RESULTS.md` /
  `LOOPED_SDFT_PHASE0_ABLATION.md`
- 코드 `model.py`(LoopedPatchTSTClassifier), `train.py`(--loop_T/--loop_inject_stab),
  `eval_dynamic_exit.py`
- 실행 `scripts/run_dual_gpu.sh`, `scripts/gen_looped_jobs.py`, `scripts/jobs_looped_ablation.txt`
- 집계 `collect_looped_results.py`
- 로그 `logs/looped_ablation_dual.log`, 메트릭 `results_json/test_metrics_loop_*_e50_s*.json`
