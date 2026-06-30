"""Phase 1/2 looped ablation jobfile 생성."""
COMMON = ("--csv RESULT1.csv --first_per_label --samples_per_label 600 "
          "--split_mode temporal --trim_head 10 --trim_tail 30 "
          "--window 10 --stride 4 --merge11 --sample_rate_hz 10.0 "
          "--arch looped_patchtst --patch_len 3 --patch_stride 1 --patch_pool mean "
          "--d_model 64 --n_heads 8 --d_ff 128 --epochs 50")

# tag -> extra args
ARMS = {
    "T2_plain":   "--loop_T 2",
    "T3_plain":   "--loop_T 3",
    "T2_stab":    "--loop_T 2 --loop_inject_stab",
    "T3_stab":    "--loop_T 3 --loop_inject_stab",
    "T6_stab":    "--loop_T 6 --loop_inject_stab",
    "T3_stab_sd": "--loop_T 3 --loop_inject_stab --multiexit_sd --multiexit_lambda 0.5 "
                  "--multiexit_temp 2.0 --multiexit_warmup 5",
}
SEEDS = [42, 1, 7]

lines = []
for tag, extra in ARMS.items():
    for s in SEEDS:
        base = f"checkpoints/best_loop_{tag}_e50_s{s}.pt"
        mj = f"results_json/test_metrics_loop_{tag}_e50_s{s}.json"
        cm = f"confusion_matrices/cm_loop_{tag}_e50_s{s}.png"
        lines.append(f"{COMMON} {extra} --seed {s} --save {base} --metrics_json {mj} --cm_png {cm}")

with open("scripts/jobs_looped_ablation.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {len(lines)} jobs to scripts/jobs_looped_ablation.txt")
