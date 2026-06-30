# Looped backbone — gated 재설계 학습 ablation 결과

작성 2026-06-29. `LOOPED_NORM_PROBE.md`에서 gated convex update가 ‖h‖ 폭발을 제거함을
확인 → 그 재설계로 **실제 학습**해 Phase 1 붕괴가 회복되는지 검증. 3-seed(42/1/7), 50ep,
dual-GPU. 비교 기준 base_L4(일반 PatchTST, F1 0.948).

## arms (model.py recur_mode/gated)

- gated + stab, T 스케일: g_T2 / g_T4 / g_T8 / g_T16
- gated, stab 유무: g_T4 vs g_T4_nostab
- gated + loop-step SDFT: g_T4_sd
- 대조(폭발): u_T4 (기존 update)

## 결과 (3-seed 평균 ± std)

| arm | acc | F1 | Δf1 vs base_L4 |
|---|---:|---:|---:|
| **base_L4 (일반 PatchTST)** | 0.9491 ± 0.0033 | **0.9479** | 기준 |
| u_T4 (update, 폭발) | 0.5833 ± 0.0393 | 0.5422 | −0.406 |
| g_T2 (gated) | 0.5903 ± 0.0204 | 0.5527 | −0.395 |
| g_T4 (gated) | 0.5301 ± 0.0164 | 0.5010 | −0.447 |
| g_T8 (gated) | 0.5000 ± 0.0397 | 0.4694 | −0.478 |
| g_T16 (gated) | 0.4630 ± 0.0740 | 0.4317 | −0.516 |
| g_T4_nostab | 0.5370 ± 0.0398 | 0.4951 | −0.453 |
| g_T4_sd | 0.5370 ± 0.0508 | 0.4935 | −0.454 |

## 핵심 발견 — ‖h‖ 폭발은 병인이 아니었다 (반증)

1. **폭발을 구조적으로 제거했는데 성능은 회복 안 됨.** gated(probe에서 ‖h‖ 1.4x 안정)가
   update(8.7x 폭발)를 **못 이긴다** (g_T4 0.501 vs u_T4 0.542). 둘 다 base_L4(0.948)에
   처참히 못 미침. → **‖h‖ 폭발은 증상이지 looped 실패의 원인이 아니었다.**

2. **진짜 병인은 과적합.** g_T4 학습 곡선: train acc 0.25→0.87(train loss 0.29)로 잘 학습하나,
   val acc는 ep10 ~0.55에서 정체, **val loss 1.16→1.64 상승**. underfit이 아니라 과적합.
   train 1320개 극소량에 looped의 effective depth가 만든 표현력이 과함.

3. **gated가 T↑일수록 여전히 악화** (g_T2 0.553 → g_T16 0.432). 폭발이 없어졌는데도 깊은
   loop가 더 나쁨 → 깊이 자체가 과적합을 키우는 것이지 폭발 때문이 아님.

4. **gated가 update보다 약간 더 나쁨.** convex update `h=(1-α)h+α·cand`가 매 step state를
   섞어 **표현 갱신을 약화**(α<1) → 같은 T라도 유효 학습 표현력이 줄어 일반화가 더 어렵다.
   안정성↔표현력 trade-off.

5. **stab/SDFT 모두 무효** (g_T4 ≈ g_T4_nostab ≈ g_T4_sd ≈ 0.50). 음의 대각도, loop-step
   self-distill도 과적합을 못 막음.

## 결론

**looped backbone 실패의 원인은 ‖h‖ 폭발(동역학적 불안정)이 아니라 소량 데이터 과적합이었다.**
gated convex update로 폭발을 깨끗이 제거한 정밀 실험이 이를 반증으로 확정. 따라서 안정화
레시피(gated / 음의 대각 / pre-norm)로는 looped를 살릴 수 없고, 근본 한계는 데이터 양이다.
Parcae가 대규모 토큰에서 성립한 것과 정반대 레짐 — "loop와 데이터를 함께 늘려라"의 역.

이로써 looped × SDFT 탐색은 **모든 안정화 변형까지 소진하며 종결**:
- Phase 0: in-network SDFT — 무익(약한 모델 안정화일 뿐).
- Phase 1: looped backbone(update) — 과적합 붕괴.
- 재설계: gated(폭발 제거) — 여전히 과적합, 회복 실패.
→ 진짜 레버는 단순 깊이(base_L4)이고, 그조차 L4에서 포화. 이 데이터엔 도메인 맞춤
backbone(hpatchfix) + teacher-soft/closed-loop 증강이 정답.

## Artifacts
- 코드 `model.py`(recur_mode=gated, _step), `train.py`(--recur_mode/--gate_init)
- probe `probe_loop_norm.py`, 집계 `collect_gated_results.py`
- 실행 `scripts/gen_gated_jobs.py`, `scripts/jobs_gated_ablation.txt`, `scripts/run_dual_gpu.sh`
- 로그 `logs/gated_ablation_dual.log`, 메트릭 `results_json/test_metrics_glp_*_e50_s*.json`
