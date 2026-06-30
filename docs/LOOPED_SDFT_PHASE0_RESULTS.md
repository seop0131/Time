# Looped × SDFT — Phase 0 결과 (multi-exit self-distillation)

작성 2026-06-29. 설계 `LOOPED_SDFT_DESIGN.md` §5 Phase 0의 실행 결과.
**looped backbone 없이**, 기존 hpatchfix(e_layers=2)의 각 encoder layer exit logit에
deep→shallow self-distillation KL을 걸어, looped 핵심 효과(중간표현 self-distill)의
방향성을 0 비용으로 검증.

## 손실

각 encoder layer exit logit z_t (t=0..T-1, T=e_layers=2). 최종 exit z_T가 anchor+teacher.

```
L = CE(z_T, y) + λ · warmup(epoch) · Σ_{t<T} τ² · KL( sg[p_τ(z_T)] || p_τ(z_t) )
```

- z_T(마지막 exit) = 기존 forward()와 동일(sanity diff=0 확인).
- teacher 신호는 stop-grad. λ=1.0, τ=2.0, warmup=5ep.

## 구현

- `model.py`: `PatchTSTClassifier._backbone_features_all_layers`(layer별 feature),
  `HybridTimeFreqClassifier.forward_multi_exit`(layer별 fusion logit, freq branch 공유).
- `train.py`: `train_epoch_multiexit_sd`, args `--multiexit_sd/_lambda/_temp/_warmup`.
- 대조군 base는 동일 코드·동일 seed로 multiexit 끄고 학습(공정 비교).

## 실험 설정

- arch `hybrid_patchtst`, freq=CNN, overlap patch(p3/s1/mean), d_model=64/h8/L2/d_ff128.
- merge11, temporal split, window=10, stride=4, 50ep, **3-seed(42/1/7)**.

## 결과 (3-seed 평균 ± std)

| arm | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **base** | **0.9282 ± 0.0087** | 0.9304 ± 0.0056 | 0.9217 ± 0.0094 | **0.9255 ± 0.0091** |
| multiexit_sd | 0.9259 ± 0.0131 | 0.9302 ± 0.0162 | 0.9192 ± 0.0143 | 0.9225 ± 0.0124 |

**Δ(sd − base): acc −0.0023, f1 −0.0031** (노이즈 범위 내, std는 오히려 증가).

## 약한 클래스 F1 (3-seed 평균)

| arm | label_7 | label_8 | label_9 | label_10 |
|---|---:|---:|---:|---:|
| base | 0.7123 | 0.8500 | 0.7455 | 0.8265 |
| multiexit_sd | 0.6889 | 0.8574 | 0.7407 | 0.8102 |

label_8만 소폭(+0.007), label_7/9/10 하락. 일관된 개선 없음.

## 해석

1. **multi-exit SD는 이 데이터에서 무효과(약한 음수).** Δf1 −0.003은 test 144 윈도우
   노이즈(±0.014) 안쪽이고, std가 커진 것(0.0091→0.0124)은 KL 항이 변동만 키웠다는 신호.
2. **누적 교훈과 일치.** self-distillation은 이 데이터의 성능 레버가 아니었고
   (`SDFT_NOTES.md`, `ONLINE_SDFT_RESULTS.md`), in-network(deep→shallow) 형태도 동일하게
   gain이 없다. teacher(최종 exit)가 student(중간 exit)에 새 정보를 주지 못하고,
   같은 backbone의 얕은 층을 깊은 층에 맞추는 것뿐이라 정보 이득이 본질적으로 없음.
3. **L=2라 exit이 2개뿐** — deep/shallow 간 표현 차이가 작아 KL이 거의 무의미했을 수 있음.
   looped backbone(T=2~3, weight-tied)은 같은 블록을 반복하므로 step 간 차이가 더 작아질
   가능성이 높다 → Phase 1에서 더 큰 gain을 기대하기 어렵다는 부정적 신호.

## 결론 / 다음 결정 (설계 §8 기준)

- **성공 기준(multiexit_sd ≥ base, 3-seed)을 충족하지 못함** (약한 음수).
- 따라서 설계 기준상 **Phase 1(looped backbone 전면 구현)은 보류**가 정공법.
- 다만 Phase 0는 SDFT 신호만 검증했고 **looped architecture의 다른 축**
  (weight-tied 깊이 확장 + Parcae injection 안정화 자체의 표현력 이득)은 검증하지 않았다.
  만약 looped를 계속 탐색한다면, 다음은 **SDFT를 빼고 순수 looped backbone(T=2,3)이
  base를 넘는지**부터 봐야 한다(loop_T2_plain / loop_T2_stab). 그게 양수면 그 위에
  SDFT를 얹는 의미가 생긴다. Phase 0 결과는 "SDFT를 먼저 얹는 경로"가 비유망함을 보여줌.

## Artifacts

- 설계 `LOOPED_SDFT_DESIGN.md`
- 스크립트 `scripts/run_multiexit_sd_gpu1.sh`, 집계 `collect_mexit_results.py`
- 로그 `logs/multiexit_sd_e50_gpu1.log`
- 체크포인트 `checkpoints/best_mexit_{base,sd}_e50_s{42,1,7}.pt`
- 메트릭 `results_json/test_metrics_mexit_{base,sd}_e50_s*.json`
