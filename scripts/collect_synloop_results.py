"""looped × 합성 teacher-soft KD 결과 집계. epoch별, base_L4/sd_init 대비."""
import json, os
import numpy as np

SEEDS = [42, 1, 7]
WEAK = ["label_7", "label_8", "label_9", "label_10"]
ARCHS = ["loopKD", "loopEMA", "hybKD"]
EPOCHS = [50, 100, 200]

REF = {"base_L4 (원본만,일반)": 0.9479, "sd_init (기존 최강 KD)": 0.9577}


def macro(pc, k):
    return float(np.mean([pc[c][k] for c in pc]))


def load(prefix):
    accs, f1s, weak = [], [], {w: [] for w in WEAK}
    for s in SEEDS:
        p = f"results_json/test_metrics_{prefix}_s{s}.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        accs.append(d["accuracy"]); f1s.append(d["f1"])
        for w in WEAK:
            if w in d["per_class"]:
                weak[w].append(d["per_class"][w]["f1"])
    return accs, f1s, weak


print("=== Main (3-seed mean ± std) ===")
print(f"{'arm':22s} {'Accuracy':18s} {'F1':18s} {'Δf1 vs base_L4':14s}")
for name, v in REF.items():
    print(f"{name:22s} {'-':18s} {v:.4f}")
print("-" * 74)
res = {}
for arch in ARCHS:
    for ep in EPOCHS:
        tag = f"{arch}_e{ep}"
        accs, f1s, weak = load(f"synloop_{tag}")
        if not accs:
            print(f"[warn] {tag}: missing"); continue
        res[tag] = dict(acc=np.mean(accs), acc_s=np.std(accs), f1=np.mean(f1s), f1_s=np.std(f1s),
                        weak={w: (np.mean(weak[w]) if weak[w] else float("nan")) for w in WEAK})
        d = res[tag]["f1"] - REF["base_L4 (원본만,일반)"]
        print(f"{tag:22s} {res[tag]['acc']:.4f} ± {res[tag]['acc_s']:.4f}   "
              f"{res[tag]['f1']:.4f} ± {res[tag]['f1_s']:.4f}   {d:+.4f}")

print("\n=== Weak-class F1 (3-seed mean) ===")
print(f"{'arm':22s} " + " ".join(f"{w:9s}" for w in WEAK))
for arch in ARCHS:
    for ep in EPOCHS:
        tag = f"{arch}_e{ep}"
        if tag in res:
            print(f"{tag:22s} " + " ".join(f"{res[tag]['weak'][w]:<9.4f}" for w in WEAK))
