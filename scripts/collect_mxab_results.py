"""multi-exit SD ablation 집계. 3-seed 평균±std + weak class + 적절한 base 대비 Δ."""
import json
import os
import numpy as np

EPOCHS = 50
SEEDS = [42, 1, 7]
WEAK = ["label_7", "label_8", "label_9", "label_10"]

# tag -> (json prefix, baseline tag for Δ)
# base_L2는 직전 실험(mexit_base) 재사용.
ARMS = [
    ("base_L2",  "mexit_base", None),
    ("base_L4",  "mxab_base_L4", None),
    ("sd_L4",    "mxab_sd_L4",  "base_L4"),
    ("lam2_L2",  "mxab_lam2_L2", "base_L2"),
    ("lam4_L2",  "mxab_lam4_L2", "base_L2"),
    ("ema_L2",   "mxab_ema_L2", "base_L2"),
    ("auxce_L2", "mxab_auxce_L2", "base_L2"),
    ("base_L6",  "mxab_base_L6", None),
    ("ema_L4",   "mxab_ema_L4", "base_L4"),
]


def macro(pc, key):
    return float(np.mean([pc[c][key] for c in pc]))


def load_arm(prefix):
    accs, f1s, precs, recs = [], [], [], []
    weak = {w: [] for w in WEAK}
    for s in SEEDS:
        path = f"results_json/test_metrics_{prefix}_e{EPOCHS}_s{s}.json"
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        accs.append(d["accuracy"]); f1s.append(d["f1"])
        pc = d["per_class"]
        precs.append(macro(pc, "precision")); recs.append(macro(pc, "recall"))
        for w in WEAK:
            if w in pc:
                weak[w].append(pc[w]["f1"])
    return accs, f1s, precs, recs, weak


res = {}
for tag, prefix, _ in ARMS:
    accs, f1s, precs, recs, weak = load_arm(prefix)
    if not accs:
        print(f"[warn] {tag}: no json ({prefix})")
        continue
    res[tag] = dict(
        acc=np.mean(accs), acc_s=np.std(accs),
        f1=np.mean(f1s), f1_s=np.std(f1s),
        prec=np.mean(precs), rec=np.mean(recs),
        weak={w: (np.mean(weak[w]) if weak[w] else float("nan")) for w in WEAK},
    )

print("\n=== Main (3-seed mean ± std) ===")
print(f"{'arm':10s} {'Accuracy':18s} {'Precision':10s} {'Recall':10s} {'F1':18s} {'Δf1 vs base':12s}")
for tag, _, base in ARMS:
    if tag not in res:
        continue
    r = res[tag]
    df1 = ""
    if base and base in res:
        df1 = f"{r['f1'] - res[base]['f1']:+.4f}"
    print(f"{tag:10s} {r['acc']:.4f} ± {r['acc_s']:.4f}   {r['prec']:.4f}    "
          f"{r['rec']:.4f}    {r['f1']:.4f} ± {r['f1_s']:.4f}   {df1:12s}")

print("\n=== Weak-class F1 (3-seed mean) ===")
print(f"{'arm':10s} " + " ".join(f"{w:10s}" for w in WEAK))
for tag, _, _ in ARMS:
    if tag not in res:
        continue
    print(f"{tag:10s} " + " ".join(f"{res[tag]['weak'][w]:<10.4f}" for w in WEAK))
