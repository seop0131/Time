"""looped × 합성 EMA-teacher relabel ablation jobfile 생성.
teacher = student EMA(self, decay 0.999). looped(gated+stab,T4). epoch{50,100,200}, 3-seed."""

COMMON = ("--csv RESULT1.csv --merge11 --samples_per_label 600 "
          "--window 10 --stride 4 --split_mode temporal --trim_head 10 --trim_tail 30 "
          "--synth synthetic_hpatchfix_teacher_1000pc.npz --mix_mode aug "
          "--arch looped_patchtst --recur_mode gated --loop_inject_stab --loop_T 4 "
          "--patch_len 3 --patch_stride 1 --patch_pool mean --d_model 64 --n_heads 8 --d_ff 128 "
          "--ema_teacher_relabel 0.999 --relabel_temp 2.0")

EPOCHS = [50, 100, 200]
SEEDS = [42, 1, 7]

lines = []
for ep in EPOCHS:
    for s in SEEDS:
        t = f"loopEMA_e{ep}"
        lines.append(f"{COMMON} --epochs {ep} --seed {s} "
                     f"--save checkpoints/best_synloop_{t}_s{s}.pt "
                     f"--metrics_json results_json/test_metrics_synloop_{t}_s{s}.json "
                     f"--cm_png confusion_matrices/cm_synloop_{t}_s{s}.png")

with open("scripts/jobs_synloop_ema.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {len(lines)} jobs to scripts/jobs_synloop_ema.txt")
