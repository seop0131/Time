# Looped backbone — per-step ‖h_t‖ 폭발 관찰 (재설계)

작성 2026-06-29. Phase 1 looped 붕괴의 동역학적 원인을 규명하기 위해, recur_mode와
injection 재설계 후 **학습 전(초기화 직후)** per-step ‖h_t‖를 관찰. `probe_loop_norm.py`.
실제 데이터 1 배치(64×10×4), torch seed 0 고정, T=16.

## 재설계 내용 (model.py LoopedPatchTSTClassifier)

기존 문제 — update에 norm을 걸어(`h = h + LN(f + W_inj·e)`) W_inj 크기가 직접 안 먹히고,
spectral_norm이 eye-init을 forward에서 덮어쓰는 충돌. → 다음으로 교체:

- **recur_mode 3종**:
  - `update`  : `h = h + LN(f(h) + W_inj·e)` (기존)
  - `prenorm` : `h = h + f(LN(h)) + W_inj·e` (state pre-norm 재귀)
  - `gated`   : `h = (1-α)·h + α·(f(LN(h)) + W_inj·e)`, α=sigmoid(학습) (convex update)
- **inject_stab**: W_inj를 **순수 음의 대각(-0.1·I)** 초기화. spectral_norm 제거(충돌 해소).
- per-step ‖h_t‖ 로깅: `forward_all_steps(x, return_norms=True)`.

## 결과 — per-step mean ‖h_t‖ (t=1..16)

| mode | stab | t1 | t2 | t4 | t8 | t16 | T16/T1 | 거동 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| update | off | 14.9 | 22.4 | 37.6 | 68.1 | 130.0 | 8.7x | **선형 폭발** |
| update | on | 15.3 | 22.8 | 37.8 | 68.2 | 129.9 | 8.5x | 폭발(stab 무력) |
| prenorm | off | 16.3 | 26.5 | 49.5 | 99.0 | 201.7 | 12.4x | **더 심한 폭발** |
| prenorm | on | 15.3 | 23.7 | 42.2 | 82.9 | 169.2 | 11.1x | 폭발 |
| **gated** | off | 8.1 | 9.3 | 11.6 | 13.1 | 13.4 | **1.6x** | **안정·수렴** |
| **gated** | on | 7.6 | 8.1 | 9.6 | 11.2 | 11.0 | **1.4x** | **안정·수렴** |

## 진단

1. **기존 `update`가 ‖h‖ 선형 폭발의 원인 확정.** 매 step norm이 ~7.6씩 단조 증가(14.9→130).
   `h = h + LN(update)`는 매 step 단위노름 벡터를 무한 누적 → ‖h‖ ∝ t. Phase 1 과적합/붕괴의
   동역학적 뿌리.

2. **inject_stab(음의 대각)은 update 모드에선 거의 무력** (8.7→8.5). 음의 대각이 빼주는 양보다
   블록 출력 누적이 압도적. "update에 작용"하는 한 효과 없음 — 재설계 지적이 정확.

3. **prenorm은 오히려 폭발 가속** (12.4x). residual 무한 누적은 동일하고, pre-norm이 블록에 더
   큰 입력을 줘 출력↑. state-norm만으론 부족.

4. **gated(convex update)만 안정.** `h=(1-α)h+α·cand`는 state가 항상 convex combination이라
   ‖h‖ 유계 수렴(~11~13에서 포화). 재설계 제안대로 **state에 작용하는 convex update가 정답.**
   gated에서는 inject_stab(음의 대각)도 추가로 norm을 낮춰 보조 효과 있음(13.4→11.0).

## 결론 / 다음

- **looped 안정화의 본질은 injection이 아니라 update 규칙**이었다. gated convex update가
  ‖h‖ 폭발을 구조적으로 제거(학습 전 동역학에서 확인).
- 이제 **gated 모드로 실제 학습 ablation**을 돌려 폭발 제거가 Phase 1 붕괴를 회복하는지 검증.

## 후속 학습 결과 (→ LOOPED_GATED_RESULTS.md) — 중요한 반증

gated로 ‖h‖ 폭발을 완전히 제거했으나 **성능은 회복되지 않았다** (g_T4 F1 0.501, base_L4 0.948).
학습 곡선은 전형적 **과적합**(train acc 0.87 / val loss 1.16→1.64 상승). 즉
**‖h‖ 폭발은 증상이지 병인이 아니었다.** 진짜 병인은 소량 데이터(train 1320) 과적합.
gated는 오히려 update보다 약간 더 나쁜데(convex가 표현 갱신 약화), 자세한 건 후속 문서 참조.

## Artifacts
- 코드 `model.py`(recur_mode/gated/_step), `train.py`(--recur_mode/--gate_init)
- probe `probe_loop_norm.py`
