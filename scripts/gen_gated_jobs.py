"""gated 재설계 looped ablation jobfile 생성."""
COMMON = ("--csv RESULT1.csv --first_per_label --samples_per_label 600 "
          "--split_mode temporal --trim_head 10 --trim_tail 30 "
          "--window 10 --stride 4 --merge11 --sample_rate_hz 10.0 "
          "--arch looped_patchtst --patch_len 3 --patch_stride 1 --patch_pool mean "
          "--d_model 64 --n_heads 8 --d_ff 128 --epochs 50")

# tag -> extra args
ARMS = {
    # gated + stab, T 스케일 (폭발 제거가 깊은 loop를 회복시키는가)
    "g_T2":        "--recur_mode gated --loop_inject_stab --loop_T 2",
    "g_T4":        "--recur_mode gated --loop_inject_stab --loop_T 4",
    "g_T8":        "--recur_mode gated --loop_inject_stab --loop_T 8",
    "g_T16":       "--recur_mode gated --loop_inject_stab --loop_T 16",
    # gated, stab 유무 (음의 대각 효과)
    "g_T4_nostab": "--recur_mode gated --loop_T 4",
    # gated + loop-step SDFT
    "g_T4_sd":     "--recur_mode gated --loop_inject_stab --loop_T 4 "
                   "--multiexit_sd --multiexit_lambda 0.5 --multiexit_temp 2.0 --multiexit_warmup 5",
    # 대조군 — 기존 폭발 모드(update) 재현
    "u_T4":        "--recur_mode update --loop_inject_stab --loop_T 4",
}
SEEDS = [42, 1, 7]

lines = []
for tag, extra in ARMS.items():
    for s in SEEDS:
        base = f"checkpoints/best_glp_{tag}_e50_s{s}.pt"
        mj = f"results_json/test_metrics_glp_{tag}_e50_s{s}.json"
        cm = f"confusion_matrices/cm_glp_{tag}_e50_s{s}.png"
        lines.append(f"{COMMON} {extra} --seed {s} --save {base} --metrics_json {mj} --cm_png {cm}")

with open("scripts/jobs_gated_ablation.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {len(lines)} jobs to scripts/jobs_gated_ablation.txt")
