"""looped × 합성 teacher-soft KD ablation jobfile 생성.
teacher = hpatchfix(best_hpatchfix_e50_s7). student arch/epoch 스윕. 3-seed."""

COMMON = ("--csv RESULT1.csv --merge11 --samples_per_label 600 "
          "--window 10 --stride 4 --split_mode temporal --trim_head 10 --trim_tail 30 "
          "--synth synthetic_hpatchfix_teacher_1000pc.npz --mix_mode aug "
          "--patch_len 3 --patch_stride 1 --patch_pool mean "
          "--d_model 64 --n_heads 8 --d_ff 128 "
          "--teacher_relabel checkpoints/best_hpatchfix_e50_s7.pt "
          "--teacher_arch hybrid_patchtst --teacher_patch_len 3 --teacher_patch_stride 1 "
          "--teacher_patch_pool mean --teacher_d_model 64 --teacher_e_layers 2 --relabel_temp 2.0")

# tag -> (arch-specific args)
ARMS = {
    # looped student (gated + stab, T4) + teacher-soft KD
    "loopKD":   "--arch looped_patchtst --recur_mode gated --loop_inject_stab --loop_T 4",
    # 일반 hybrid_patchtst student + teacher-soft KD (= sd 계열, 공정 비교 대조)
    "hybKD":    "--arch hybrid_patchtst --e_layers 2",
}
EPOCHS = [50, 100, 200]
SEEDS = [42, 1, 7]

lines = []
for tag, extra in ARMS.items():
    for ep in EPOCHS:
        for s in SEEDS:
            t = f"{tag}_e{ep}"
            base = f"checkpoints/best_synloop_{t}_s{s}.pt"
            mj = f"results_json/test_metrics_synloop_{t}_s{s}.json"
            cm = f"confusion_matrices/cm_synloop_{t}_s{s}.png"
            lines.append(f"{COMMON} {extra} --epochs {ep} --seed {s} "
                         f"--save {base} --metrics_json {mj} --cm_png {cm}")

with open("scripts/jobs_synloop.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {len(lines)} jobs to scripts/jobs_synloop.txt")
