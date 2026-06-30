# Looped × 합성 teacher-soft KD 결과 (원래 의도대로 — diffusion 합성 + KD)

작성 2026-06-29. **기존 looped 실험들이 합성 KD 없이 원본(train 1320)만 썼던 누락을 교정.**
이 프로젝트의 핵심 레버는 diffusion 합성 + teacher-soft KD였으므로, looped student도
그 레시피로 학습해 base_L4(0.948) / sd_init(0.958)를 넘는지 검증.

## 셋업

- 데이터: 원본 1320(hard CE) + 합성 11000(`synthetic_hpatchfix_teacher_1000pc.npz`,
  teacher soft-label KL, T=2) = 12320. (train_with_synth.py 경로 B)
- teacher 2종: **frozen hpatchfix**(`best_hpatchfix_e50_s7.pt`, 0.958) vs
  **student EMA(self, decay 0.999)**.
- student arch 비교: **looped_patchtst(gated+stab, T4)** vs **hybrid_patchtst(일반, L2)**.
- epoch 스윕: 50 / 100 / 200. 3-seed(42/1/7). dual-GPU.
- 구현: `train_with_synth.py`에 looped_patchtst + EMA teacher relabel 지원 추가.

## 결과 (3-seed 평균 ± std)

| arm | acc | F1 | Δf1 vs base_L4 |
|---|---:|---:|---:|
| (참고) base_L4 원본만, 일반 | — | 0.9479 | — |
| (참고) sd_init 기존 최강 KD | — | 0.9577 | — |
| **hybKD_e50** (일반+KD) | 0.9537 | 0.9525 | +0.005 |
| **hybKD_e100** (일반+KD) | **0.9560** | **0.9548** | **+0.007** |
| hybKD_e200 (일반+KD) | 0.9491 | 0.9478 | −0.000 |
| loopKD_e50 (hpatchfix teacher) | 0.6551 | 0.6135 | −0.334 |
| loopKD_e100 | 0.7037 | 0.6826 | −0.265 |
| loopKD_e200 | 0.7454 ± 0.069 | 0.7201 | −0.228 |
| loopEMA_e50 (self EMA teacher) | 0.2986 | 0.2391 | −0.709 |
| loopEMA_e100 | 0.4282 ± 0.110 | 0.3797 | −0.568 |
| loopEMA_e200 | 0.4884 ± 0.096 | 0.4553 | −0.493 |

## 핵심 발견

1. **합성 KD가 looped를 분명히 끌어올렸다 (질문의 답 = 맞다).** 원본만 쓴 gated 실험
   (F1 ~0.50)보다 합성 KD가 훨씬 낫고(0.61→0.72), **epoch↑일수록 계속 회복**
   (e50 0.614 → e100 0.683 → e200 0.720). 데이터 11배 증가가 과적합을 완화 —
   "looped 실패의 원인은 데이터 양"이라는 진단(`LOOPED_GATED_RESULTS.md`)을 확증.

2. **그래도 looped는 일반 backbone에 한참 못 미침.** 같은 합성 KD·같은 teacher인데
   arch만 다른데 격차 0.23 (loopKD 0.72 vs hybKD 0.955). weight-tied looped의 구조적
   표현력 부족(짧은 윈도우·소량 레짐)이 데이터를 늘려도 남는다.

3. **일반 PatchTST + 합성 KD는 base_L4 초과, sd_init 근접.** hybKD_e100 0.9548 ≈ sd_init
   0.9577. 프로젝트 최강 레시피 재확인.

4. **epoch 효과가 arch별로 반대.** looped는 길수록↑(데이터 많아 더 학습 여지),
   일반은 e100 최적·e200 과적합으로 소폭↓. → looped는 데이터/epoch를 더 주면 더 오르지만
   (Parcae scaling 정신), base를 따라잡으려면 격차가 너무 크다.

5. 약한 클래스: loopKD도 epoch↑로 label_8 회복(0.05→0.86), 그러나 label_7은 계속 약함
   (0.12→0.28). 일반 hybKD는 모든 약한 클래스 안정(0.80~0.92).

6. **frozen hpatchfix teacher ≫ student EMA teacher (큰 차이).** 같은 looped student인데
   teacher만 바꿔도 격차가 크다 — loopKD(hpatchfix) 0.61~0.72 vs loopEMA(self) 0.24~0.46.
   EMA는 std도 큼(±0.10), 약한 클래스 다수 0. **외부의 강한 teacher가 핵심**이고, 자기증류
   EMA는 looped에서도 무익(Phase 0/1 EMA 교훈과 일관). EMA도 epoch↑로 오르나(0.24→0.46)
   여전히 최악. → looped엔 강한 외부 teacher(hpatchfix)가 필수지만 그래도 일반 backbone 못 넘음.

## 결론

**합성 KD는 looped의 과적합을 완화하지만(데이터 양 진단 확증), looped 자체의 구조적 한계로
일반 PatchTST를 넘지 못한다.** 즉 "원본만 써서 실패했다"는 지적은 옳았고 합성 KD로 크게
개선됐으나, 결론은 바뀌지 않는다 — 이 데이터(window=10, 소량)에선 **일반 overlap-PatchTST +
합성 teacher-soft KD(hybKD/sd_init ≈ 0.955)가 정답**이고 looped는 부적합.

looped를 끝까지 밀려면 데이터·epoch를 더 키우는 방향(Parcae scaling)인데, 격차(0.23)가
커서 실용 가망은 낮다. looped × SDFT/KD 탐색은 여기서 종결.

## Artifacts
- 코드 `train_with_synth.py`(looped 지원), `model.py`
- 생성 `scripts/gen_synloop_jobs.py`, `scripts/jobs_synloop.txt`, `scripts/run_dual_gpu.sh`(PYSCRIPT)
- 집계 `collect_synloop_results.py`, 로그 `logs/synloop_dual.log`
- 메트릭 `results_json/test_metrics_synloop_*_s*.json`
