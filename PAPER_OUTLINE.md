# Paper Outline — In-Network Self-Distillation for Data-Scarce Short-Window Time-Series Classification: An Empirical Study

작성 2026-06-29. 프레이밍 = **empirical study (B)**, 타깃 = 정규 논문 8~10p.
in-network SDFT를 주제로, "frontier 자기증류/looped 기법을 소량·단윈도우 시계열에 이식 →
언제 되고 안 되는가"를 통제된 ablation으로 규명. 모든 수치는 본 repo 3-seed(42/1/7) 결과.

---

## 제목 후보
- **"When Does In-Network Self-Distillation Help? An Empirical Study on Data-Scarce
  Short-Window Time-Series Classification"**
- 대안: "Self-Distillation and Looped Architectures Meet Tiny Time-Series: A Controlled Study"

## 한 문장 기여 (thesis)
소량·단윈도우 시계열에서 in-network/looped 자기증류 기법은 **약한 backbone을 안정화할 뿐
강한 backbone을 못 넘으며**, 진짜 이득은 (i) overlapping-patch로 token starvation 해소,
(ii) diffusion 합성 + 외부 teacher-soft KD에서 온다. 자기증류의 효용은 backbone 강도와
데이터 양에 대한 **조건부**임을 통제 실험으로 규명.

---

## Abstract (요지)
- 문제: 소량(클래스당 1분)·단윈도우(W=10) 4채널 시계열 11-class 분류.
- 한 일: PatchTST 위에 in-network SDFT, looped(LoopViT/Parcae) backbone, 합성 KD를
  체계적 ablation(3-seed, temporal split).
- 발견: (1) overlapping patch가 token starvation을 풀어 0.79→0.94. (2) in-network SDFT는
  약한 backbone만 안정화, 강한 backbone(L4)엔 무익/유해. (3) looped는 ‖h‖ 폭발이 아니라
  소량 데이터 과적합으로 실패(gated convex update로 폭발 제거해도 회복 안 됨). (4) 진짜
  레버는 합성 teacher-soft KD(0.955, foundation 동급).
- 메시지: 자기증류/looped의 효용은 조건부. 소량 레짐엔 도메인 backbone + 합성 KD가 정답.

---

## 1. Introduction
- 동기: 산업 센서(배터리 4채널) 동작분류 — 라벨·데이터 극소량, 윈도우 짧음.
- frontier 추세: self-distillation(SDFT), looped/recurrent depth(LoopViT, Parcae),
  KD. 이들이 소량·단윈도우 시계열에 통하는가? = 본 연구 질문.
- 기여:
  1. 단윈도우 PatchTST의 token starvation 정량화 + overlapping patch 해법.
  2. in-network SDFT(multi-exit deep→shallow)를 분류에 정식화하고 한계 규명.
  3. looped backbone 실패의 원인을 ‖h‖ 동역학(probe) + 과적합으로 분리(gated 반증).
  4. 합성 teacher-soft KD가 유일한 일관 레버임을 통제 비교로 확립.
- 정직성: 본 논문은 다수 negative를 포함하며, 그것이 "frontier 기법 무지성 적용" 경계로서 기여.

## 2. Related Work
- PatchTST / 단윈도우 시계열 분류, token starvation.
- Knowledge distillation, self-distillation(BYOT, deep supervision), SDFT(sail-sg),
  Mean-Teacher/EMA consistency.
- Looped/weight-tied recurrence: Universal Transformer, LoopViT, **Parcae(scaling law,
  injection 안정화)**.
- Diffusion 기반 시계열 합성 + KD. data-scarce regime.

## 3. Problem Setup & Backbone (positive 토대)
- 데이터: 4채널(VBAT/IBAT/ICHG/ILOAD), 10Hz, W=10, 11-class(merge11),
  train 1320/val 144/test 144, temporal split. **3-seed 필수**(test 144 → 단일 seed ±0.014).
- §3.1 Token starvation 진단: non-overlap patch(p5/s5)→토큰 2개, random 0.97 vs temporal 0.79.
- §3.2 해법 overlapping patch(p3/s1)→토큰 8개 + FFT branch = hpatchfix.
  | 단계 | F1 |
  |---|---:|
  | vanilla PatchTST | 0.81 |
  | overlap (no-freq) | 0.84 |
  | hpatchfix (overlap+FFT) | 0.94 |
- 깊이 효과: base_L2 0.926 → **base_L4 0.948** → base_L6 0.945(포화). ← 강한 backbone 기준선.

## 4. Method — In-Network Self-Distillation (주제)
- §4.1 정식화: encoder layer/loop step별 exit logit z_t. 최종 z_T가 anchor+teacher.
  L = CE(z_T,y) + λ·Σ_{t<T} T²·KL(sg[p_τ(z_T)] ‖ p_τ(z_t)).
- §4.2 변형 축: teacher(stop-grad final / EMA / 중간 aux CE), λ, exit 수(L), temperature.
- §4.3 looped 확장: weight-tied 블록 T번 재귀(LoopViT/Parcae식), injection,
  recur_mode(update/prenorm/gated), Parcae 안정화(음의 대각).
- §4.4 합성 KD 결합: real(hard CE) + synth(외부 teacher soft-label KL). diffusion 생성.

## 5. Experiments (ablation 본체 — 가진 결과 그대로)

### 5.1 In-network SDFT는 약한 backbone만 안정화 (조건부)
| arm | L | F1 | Δ vs 동일-L base |
|---|---:|---:|---:|
| base_L2 | 2 | 0.9255 | — |
| ema_L2 | 2 | 0.9454 | **+0.020** |
| base_L4 | 4 | 0.9479 | — |
| ema_L4 | 4 | 0.9233 | **−0.025** |
| sd_L4 (강KL) | 4 | 0.840 | −0.108 (collapse) |
| lam4_L2 | 2 | 0.793 | −0.133 (collapse) |
- 메시지: EMA gain은 안정화이지 distillation 이득 아님(L4서 역전). 강 KL은 collapse.

### 5.2 Looped 실패 원인 분리 — 폭발이 아니라 과적합
- per-step ‖h_t‖ probe(학습 전): update 8.7x / prenorm 12.4x **폭발**, gated **1.4x 안정**.
- 그러나 학습: gated(폭발 제거)도 회복 안 됨(g_T2 0.553, T↑일수록 악화 0.43).
  학습곡선 = 과적합(train↑ val loss↑). → **‖h‖ 폭발은 증상, 병인은 소량 과적합.**

### 5.3 합성 KD — 데이터를 늘리면 looped 회복(but 부족), 일반은 SOTA
| student/teacher | e50 | e100 | e200 |
|---|---:|---:|---:|
| hybKD (일반/hpatchfix) | 0.953 | **0.955** | 0.948 |
| loopKD (looped/hpatchfix) | 0.614 | 0.683 | 0.720 |
| loopEMA (looped/self-EMA) | 0.239 | 0.380 | 0.455 |
- looped는 데이터·epoch↑로 회복(과적합 진단 확증) but 일반 backbone 못 넘음(−0.23).
- 외부 강 teacher(hpatchfix) ≫ self-EMA. 일반+합성 KD = sd_init(0.958) 동급.

### 5.4 Foundation 비교 (positive 마무리)
- frozen TS foundation(Utica)+probe 0.958 ≈ 우리 일반+합성 KD 0.955 → 사전학습 없이 동급.

## 6. Analysis / Discussion
- "언제 self-distillation이 되나" 조건표:
  | 조건 | self-distill 효용 |
  |---|---|
  | 약한 backbone, 소량 | 안정화로 약간 ↑ (regression-to-mean) |
  | 강한 backbone | 무익/유해 |
  | 강 KL / 깊은 loop | collapse |
- Parcae scaling law("loop와 데이터를 함께 늘려라")의 역 — 소량이라 loop 늘릴 여력 없음.
- 진짜 레버 = backbone(overlap patch) + 외부 teacher-soft 합성 KD.

## 7. Limitations
- 단일 데이터셋, test 144(작음) → 다른 도메인/public benchmark 일반화 필요.
- in-network SDFT positive 영역(약 backbone) 검증은 제한적.

## 8. Conclusion
- 자기증류/looped는 frontier지만 소량·단윈도우엔 조건부. 무지성 적용 경계.
- 권고 레시피: overlap-patch backbone + diffusion 합성 teacher-soft KD.

---

## 그림/표 계획
- Fig1 token starvation(패치 2 vs 8) 다이어그램.
- Fig2 in-network SDFT / looped 구조도.
- Fig3 per-step ‖h_t‖ probe (update/prenorm/gated × T).
- Fig4 looped 학습곡선(train↑ val↓ 과적합).
- Tab1 backbone ladder(0.81→0.94→0.948).
- Tab2 in-network SDFT ablation(5.1).
- Tab3 합성 KD epoch 스윕(5.3).
- Tab4 foundation 비교.

## 근거 문서(repo)
- backbone: PATCHTST_NOFREQ_BASELINE / UTICA_PATCH_ABLATION / HPATCHFIX_TEACHER_SD
- in-network SDFT: LOOPED_SDFT_PHASE0_RESULTS / _ABLATION
- looped: LOOPED_SDFT_DESIGN / _PHASE12_RESULTS / NORM_PROBE / GATED_RESULTS
- 합성 KD: LOOPED_SYNTH_KD_RESULTS, CONFAWARE_SD / SELECTION_NOTES / CLOSED_LOOP
- foundation: UTICA_PROBE_RESULTS. 전체: METHODS_MASTER.md
