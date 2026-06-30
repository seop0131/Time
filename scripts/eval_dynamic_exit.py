"""
Phase 2 — Dynamic Exit 평가.
학습된 looped_patchtst 체크포인트를 로드해 test set에서 entropy threshold별
평균 사용 loop step과 accuracy를 측정한다. (조기종료 효율 vs 정확도 trade-off)

사용: python eval_dynamic_exit.py --ckpt checkpoints/best_loop_T6_stab_e50_s7.pt --loop_T 6
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import make_splits_first_per_label
from model import build_model


def _safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--loop_T", type=int, default=6)
    ap.add_argument("--inject_stab", action="store_true", default=True)
    ap.add_argument("--csv", default="RESULT1.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_ds = make_splits_first_per_label(
        args.csv, samples_per_label=600, window=10, stride=4,
        train_ratio=0.7, val_ratio=0.15, binary=False, seed=args.seed,
        split_mode="temporal", trim_head=10, trim_tail=30, merge11=True,
    )
    num_classes = int(test_ds.targets.max()) + 1
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = build_model(num_classes=num_classes, arch="looped_patchtst",
                        seq_len=10, patch_len=3, stride=1, d_model=64, n_heads=8,
                        d_ff=128, loop_T=args.loop_T, inject_stab=args.inject_stab, pool="mean").to(device)
    model.load_state_dict(_safe_load(args.ckpt, device))
    model.eval()

    print(f"ckpt={args.ckpt}  T={args.loop_T}")
    print(f"{'entropy_thr':>12s} {'mean_steps':>11s} {'accuracy':>9s}")
    for thr in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 99.0]:  # 0=항상 끝까지, 99=항상 첫 step
        correct = total = 0
        steps_all = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits, used = model.forward_dynamic_exit(x, entropy_thresh=thr)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
                steps_all.append(used.float().cpu())
        ms = torch.cat(steps_all).mean().item()
        print(f"{thr:12.2f} {ms:11.2f} {correct / total:9.4f}")


if __name__ == "__main__":
    main()
