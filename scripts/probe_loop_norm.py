"""
Looped backbone의 per-step ‖h_t‖ 폭발 관찰 (학습 전, 초기화 직후).
recur_mode × inject_stab × T 조합으로 실제 데이터 한 배치를 흘려 step별 평균 ‖h_t‖를 출력.
폭발(증가) vs 안정(수렴) 여부를 먼저 본다.

사용: python probe_loop_norm.py
"""
import torch
from torch.utils.data import DataLoader
from dataset import make_splits_first_per_label
from model import build_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    train_ds, _, _ = make_splits_first_per_label(
        "RESULT1.csv", samples_per_label=600, window=10, stride=4,
        train_ratio=0.7, val_ratio=0.15, binary=False, seed=42,
        split_mode="temporal", trim_head=10, trim_tail=30, merge11=True,
    )
    num_classes = int(train_ds.targets.max()) + 1
    x, _ = next(iter(DataLoader(train_ds, batch_size=64, shuffle=False)))
    x = x.to(device)

    T_MAX = 16
    print(f"batch={tuple(x.shape)}  num_classes={num_classes}  (학습 전, 초기화 직후)")
    print(f"{'mode':9s} {'stab':5s} | per-step mean ||h_t||  (t=1..{T_MAX})")
    print("-" * 90)
    for mode in ["update", "prenorm", "gated"]:
        for stab in [False, True]:
            torch.manual_seed(0)
            m = build_model(num_classes=num_classes, arch="looped_patchtst",
                            seq_len=10, patch_len=3, stride=1, d_model=64, n_heads=8,
                            d_ff=128, loop_T=T_MAX, inject_stab=stab,
                            recur_mode=mode, gate_init=0.5, pool="mean").to(device)
            m.eval()
            with torch.no_grad():
                _, norms = m.forward_all_steps(x, return_norms=True)
            # 폭발 지표: 마지막/첫 비율
            ratio = norms[-1] / (norms[0] + 1e-8)
            shown = "  ".join(f"{n:6.2f}" for n in norms)
            print(f"{mode:9s} {str(stab):5s} | {shown}   (T16/T1={ratio:5.1f}x)")
    print("\n해석: ratio가 크면(↑) 폭발, ~1 또는 감소면 안정.")


if __name__ == "__main__":
    main()
