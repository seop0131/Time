"""Phase 1/2 looped ablation 집계. Phase 0 base들과 함께 비교."""
import json, os
import numpy as np

EPOCHS = 50
SEEDS = [42, 1, 7]
WEAK = ["label_7", "label_8", "label_9", "label_10"]

# (label, json prefix)
ARMS = [
    ("base_L2 (Ph0)",  "mexit_base"),
    ("base_L4 (Ph0)",  "mxab_base_L4"),
    ("loop_T2_plain",  "loop_T2_plain"),
    ("loop_T3_plain",  "loop_T3_plain"),
    ("loop_T2_stab",   "loop_T2_stab"),
    ("loop_T3_stab",   "loop_T3_stab"),
    ("loop_T6_stab",   "loop_T6_stab"),
    ("loop_T3_stab_sd","loop_T3_stab_sd"),
]


def macro(pc, k):
    return float(np.mean([pc[c][k] for c in pc]))


def load(prefix):
    accs, f1s, precs, recs, weak = [], [], [], [], {w: [] for w in WEAK}
    for s in SEEDS:
        p = f"results_json/test_metrics_{prefix}_e{EPOCHS}_s{s}.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        accs.append(d["accuracy"]); f1s.append(d["f1"])
        pc = d["per_class"]; precs.append(macro(pc, "precision")); recs.append(macro(pc, "recall"))
        for w in WEAK:
            if w in pc: weak[w].append(pc[w]["f1"])
    return accs, f1s, precs, recs, weak


res = {}
for label, prefix in ARMS:
    accs, f1s, precs, recs, weak = load(prefix)
    if not accs:
        print(f"[warn] {label}: missing ({prefix})"); continue
    res[label] = dict(acc=np.mean(accs), acc_s=np.std(accs), f1=np.mean(f1s), f1_s=np.std(f1s),
                      prec=np.mean(precs), rec=np.mean(recs),
                      weak={w: (np.mean(weak[w]) if weak[w] else float("nan")) for w in WEAK})

base4 = res.get("base_L4 (Ph0)", {}).get("f1")
print("\n=== Main (3-seed mean ± std) ===")
print(f"{'arm':18s} {'Accuracy':18s} {'Prec':8s} {'Rec':8s} {'F1':18s} {'Δf1 vs base_L4':14s}")
for label, _ in ARMS:
    if label not in res: continue
    r = res[label]
    dv = f"{r['f1'] - base4:+.4f}" if base4 is not None else ""
    print(f"{label:18s} {r['acc']:.4f} ± {r['acc_s']:.4f}   {r['prec']:.4f}  {r['rec']:.4f}  "
          f"{r['f1']:.4f} ± {r['f1_s']:.4f}   {dv:14s}")

print("\n=== Weak-class F1 (3-seed mean) ===")
print(f"{'arm':18s} " + " ".join(f"{w:10s}" for w in WEAK))
for label, _ in ARMS:
    if label not in res: continue
    print(f"{label:18s} " + " ".join(f"{res[label]['weak'][w]:<10.4f}" for w in WEAK))
