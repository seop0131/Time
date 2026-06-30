"""
원본 + diffusion 합성 데이터로 분류 모델 학습.

사용 예:
  python train_with_synth.py \
    --csv RESULT1.csv --merge11 \
    --synth synthetic_windows.npz \
    --mix_mode aug \
    --arch hybrid_cnn_bilstm --window 10 \
    --save checkpoints/best_aug.pt \
    --metrics_json results_json/test_aug.json \
    --cm_png confusion_matrices/cm_aug.png

mix_mode:
  - only:  합성 데이터만 학습 (sanity check)
  - aug:   원본 train + 합성 (확장)
"""

import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from tqdm import tqdm

from dataset import make_splits_first_per_label, SlidingWindowDataset
from model import build_model
from metrics import compute_metrics, print_metrics, save_confusion_matrix, save_metrics_json


class SyntheticDataset(Dataset):
    """npz로 저장된 합성 윈도우 X(N,W,C), Y(N,)를 받아 SlidingWindowDataset 호환."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.samples = X.astype(np.float32)
        self.targets = Y.astype(np.int64)
        self.window = X.shape[1]
        self.transform = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx])
        if self.transform is not None:
            x = self.transform(x)
        return x, torch.tensor(self.targets[idx])


class FlaggedDataset(Dataset):
    """기존 dataset을 감싸 (x, y, is_synth) 로 반환. real=0, synth=1."""

    def __init__(self, base, is_synth):
        self.base = base
        self.is_synth = int(is_synth)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, self.is_synth


def _safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _ema_update(teacher, student, decay):
    """teacher = decay·teacher + (1-decay)·student (weight + buffer)."""
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data.mul_(decay).add_(s.data, alpha=1.0 - decay)
    for t, s in zip(teacher.buffers(), student.buffers()):
        t.data.copy_(s.data)


def make_model_kwargs(
    arch, window, sample_rate_hz, patch_len, patch_stride, patch_pool,
    d_model, n_heads, e_layers, d_ff,
    loop_T=3, inject_stab=False, recur_mode="update", gate_init=0.5,
):
    kwargs = {}
    pool = patch_pool
    if arch == "cross_patchtst" and pool == "flatten":
        pool = "cls"
    if arch == "looped_patchtst":
        return dict(
            seq_len=window, patch_len=patch_len, stride=patch_stride,
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            loop_T=loop_T, inject_stab=inject_stab,
            recur_mode=recur_mode, gate_init=gate_init,
            pool=("mean" if pool == "cls" else pool),
        )
    if arch in {"patchtst", "hybrid_patchtst", "cross_patchtst"}:
        patch_kwargs = dict(
            seq_len=window,
            patch_len=patch_len,
            stride=patch_stride,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            pool=pool,
        )
        if arch in {"patchtst", "cross_patchtst"}:
            kwargs.update(patch_kwargs)
        else:
            kwargs["window"] = window
            kwargs["sample_rate_hz"] = sample_rate_hz
            kwargs["patchtst_kwargs"] = patch_kwargs
    elif arch.startswith("hybrid_"):
        kwargs["window"] = window
        kwargs["sample_rate_hz"] = sample_rate_hz
    return kwargs


def distill_kl(student_logits, teacher_logits, temperature):
    p_t = torch.softmax(teacher_logits / temperature, dim=-1)
    log_p_s = torch.log_softmax(student_logits / temperature, dim=-1)
    return (p_t * (torch.log(p_t + 1e-8) - log_p_s)).sum(-1) * (temperature ** 2)


def train_epoch_relabel(model, teacher, loader, optimizer, criterion, device,
                        epoch, epochs, temperature, conf_min=0.0,
                        conf_weight="none", synth_class_weights=None,
                        real_kd_alpha=1.0):
    """
    SDFT relabel (경로 B): real 샘플은 hard-label CE, synth 샘플은 teacher soft-label KL.

      real(is_synth=0) : CE(student(x), y), optionally CE+teacher KL
      synth(is_synth=1): T²·KL(teacher(x)^T || student(x)^T)   ← hard label 미사용

    선택 옵션:
      - conf_min: teacher confidence가 낮은 합성 샘플은 KL에서 제외
      - conf_weight: teacher confidence/entropy로 합성 KL sample weight 조절
      - synth_class_weights: confusion-aware class weight로 약한 class synthetic KL 강조

    원전 SDFT의 "정답을 모델 분포로 재작성"을 분류로 옮긴 형태. teacher는 freeze real-data 모델.
    loader는 (x, y, is_synth) batch를 줘야 함 (FlaggedDataset).
    """
    model.train()
    teacher.eval()
    total_loss = correct = total = 0
    n_ce = n_kl = n_drop = n_real_kd = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{epochs} [relabel]", leave=False)
    for x, y, is_synth in pbar:
        x, y, is_synth = x.to(device), y.to(device), is_synth.to(device)
        optimizer.zero_grad()
        logits = model(x)

        real_mask = is_synth == 0
        synth_mask = ~real_mask
        loss = x.new_zeros(())
        # real: hard CE
        if real_mask.any():
            ce = criterion(logits[real_mask], y[real_mask])
            if real_kd_alpha < 1.0:
                with torch.no_grad():
                    real_teacher_logits = teacher(x[real_mask])
                real_kl = distill_kl(logits[real_mask], real_teacher_logits, temperature).mean()
                real_loss = real_kd_alpha * ce + (1.0 - real_kd_alpha) * real_kl
                n_real_kd += int(real_mask.sum())
            else:
                real_loss = ce
            loss = loss + real_loss * real_mask.sum() / len(y)
            n_ce += int(real_mask.sum())
        # synth: teacher soft-label KL
        if synth_mask.any():
            x_syn = x[synth_mask]
            y_syn = y[synth_mask]
            logits_syn = logits[synth_mask]
            with torch.no_grad():
                teacher_logits = teacher(x_syn)
                p_raw = torch.softmax(teacher_logits, dim=-1)
                conf = p_raw.max(dim=-1).values

            keep = conf >= conf_min
            n_drop += int((~keep).sum())
            if keep.any():
                kl_each = distill_kl(logits_syn[keep], teacher_logits[keep], temperature)

                weights = torch.ones_like(kl_each)
                if conf_weight == "max":
                    weights = weights * conf[keep]
                elif conf_weight == "entropy":
                    entropy = -(p_raw[keep] * torch.log(p_raw[keep] + 1e-8)).sum(dim=-1)
                    entropy_weight = 1.0 - entropy / math.log(p_raw.size(-1))
                    weights = weights * entropy_weight.clamp_min(0.0)

                if synth_class_weights is not None:
                    weights = weights * synth_class_weights[y_syn[keep]]

                kl = (kl_each * weights).sum() / (weights.sum() + 1e-8)
                loss = loss + kl * keep.sum() / len(y)
                n_kl += int(keep.sum())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
        pbar.set_postfix(
            loss=f"{total_loss/total:.4f}",
            acc=f"{correct/total:.4f}",
            kl=n_kl,
            drop=n_drop,
            rkd=n_real_kd,
        )
    return total_loss / total, correct / total


def train_epoch(model, loader, optimizer, criterion, device, epoch, epochs):
    model.train()
    total_loss = correct = total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{epochs}", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, desc="eval"):
    model.eval()
    total_loss = correct = total = 0
    preds_all, labels_all = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(1)
        total_loss += loss.item() * len(y)
        correct += (preds == y).sum().item()
        total += len(y)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(y.cpu().numpy())
    return total_loss / total, correct / total, np.array(preds_all), np.array(labels_all)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 원본 split (val/test는 항상 원본 사용)
    train_ds, val_ds, test_ds = make_splits_first_per_label(
        args.csv,
        samples_per_label=args.samples_per_label,
        window=args.window,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        binary=False,
        seed=args.seed,
        split_mode=args.split_mode,
        trim_head=args.trim_head,
        trim_tail=args.trim_tail,
        merge11=args.merge11,
    )

    # 합성 데이터 로드
    npz = np.load(args.synth)
    X_syn, Y_syn = npz["X"], npz["Y"]
    print(f"Loaded synth: X={X_syn.shape}  Y={Y_syn.shape}  classes={sorted(set(Y_syn.tolist()))}")
    syn_ds = SyntheticDataset(X_syn, Y_syn)

    if args.mix_mode == "only":
        train_use = syn_ds
        print(f"[only] synth-only train: {len(train_use)}")
    elif args.mix_mode == "aug":
        if args.teacher_relabel or args.ema_teacher_relabel > 0:
            # 경로 B(SDFT relabel): real/synth를 flag로 구분해 합침 (frozen 또는 EMA teacher)
            train_use = ConcatDataset([FlaggedDataset(train_ds, 0), FlaggedDataset(syn_ds, 1)])
            print(f"[aug+relabel] orig({len(train_ds)}, hard) + synth({len(syn_ds)}, teacher-soft) = {len(train_use)}")
        else:
            train_use = ConcatDataset([train_ds, syn_ds])
            print(f"[aug] orig({len(train_ds)}) + synth({len(syn_ds)}) = {len(train_use)}")
    else:
        raise ValueError(args.mix_mode)

    train_loader = DataLoader(train_use, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    # 클래스 분포 (train 기준)
    if args.mix_mode == "only":
        targets_for_weight = Y_syn
    elif args.ce_weight_real_only:
        targets_for_weight = train_ds.targets
    else:
        targets_for_weight = np.concatenate([train_ds.targets, Y_syn])
    label_counts = np.bincount(targets_for_weight)
    class_weights = 1.0 / (label_counts + 1e-6)
    class_weights = torch.tensor(
        class_weights / class_weights.sum() * len(label_counts),
        dtype=torch.float32,
    ).to(device)

    num_classes = int(targets_for_weight.max()) + 1
    model_kwargs = make_model_kwargs(
        args.arch, args.window, args.sample_rate_hz,
        args.patch_len, args.patch_stride, args.patch_pool,
        args.d_model, args.n_heads, args.e_layers, args.d_ff,
        loop_T=args.loop_T, inject_stab=args.loop_inject_stab,
        recur_mode=args.recur_mode, gate_init=args.gate_init,
    )

    model = build_model(num_classes=num_classes, arch=args.arch, **model_kwargs).to(device)
    if args.init_from:
        model.load_state_dict(_safe_torch_load(args.init_from, map_location=device))
        print(f"Initialized student from {args.init_from} (fine-tune mode)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 경로 B-EMA: student EMA를 relabel teacher로 (frozen teacher 대신 자기증류)
    teacher_model = None
    ema_relabel = bool(args.ema_teacher_relabel > 0)
    if ema_relabel:
        teacher_model = build_model(num_classes=num_classes, arch=args.arch, **model_kwargs).to(device)
        teacher_model.load_state_dict(model.state_dict())
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        teacher_model.eval()
        print(f"[relabel] EMA teacher (self, decay={args.ema_teacher_relabel}) — "
              f"synth는 student-EMA soft-label, T={args.relabel_temp}")
    # 경로 B: teacher relabel용 freeze teacher 로드
    elif args.teacher_relabel:
        teacher_arch = args.teacher_arch or args.arch
        teacher_kwargs = make_model_kwargs(
            teacher_arch,
            args.teacher_window or args.window,
            args.teacher_sample_rate_hz or args.sample_rate_hz,
            args.teacher_patch_len if args.teacher_patch_len is not None else args.patch_len,
            args.teacher_patch_stride if args.teacher_patch_stride is not None else args.patch_stride,
            args.teacher_patch_pool or args.patch_pool,
            args.teacher_d_model if args.teacher_d_model is not None else args.d_model,
            args.teacher_n_heads if args.teacher_n_heads is not None else args.n_heads,
            args.teacher_e_layers if args.teacher_e_layers is not None else args.e_layers,
            args.teacher_d_ff if args.teacher_d_ff is not None else args.d_ff,
        )
        teacher_model = build_model(num_classes=num_classes, arch=teacher_arch, **teacher_kwargs).to(device)
        teacher_model.load_state_dict(_safe_torch_load(args.teacher_relabel, map_location=device))
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        teacher_model.eval()
        print(f"[relabel] teacher loaded: {args.teacher_relabel}  "
              f"(arch={teacher_arch}, synth는 teacher soft-label, T={args.relabel_temp})")
        if args.real_kd_alpha < 1.0:
            print(f"[relabel] real KD enabled: CE alpha={args.real_kd_alpha}, "
                  f"KL weight={1.0 - args.real_kd_alpha}")

    synth_class_weights = None
    if args.relabel_class_weights_json:
        with open(args.relabel_class_weights_json) as f:
            raw_weights = json.load(f)
        weights = torch.ones(num_classes, dtype=torch.float32, device=device)
        for k, v in raw_weights.items():
            cls = int(k)
            if 0 <= cls < num_classes:
                weights[cls] = float(v)
        synth_class_weights = weights
        print(f"[relabel] synth class weights: {weights.detach().cpu().tolist()}")
    if args.relabel_conf_min > 0 or args.relabel_conf_weight != "none":
        print(f"[relabel] confidence control: min={args.relabel_conf_min}, weight={args.relabel_conf_weight}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        if teacher_model is not None:
            train_loss, train_acc = train_epoch_relabel(
                model, teacher_model, train_loader, optimizer, criterion, device,
                epoch, args.epochs, temperature=args.relabel_temp,
                conf_min=args.relabel_conf_min,
                conf_weight=args.relabel_conf_weight,
                synth_class_weights=synth_class_weights,
                real_kd_alpha=args.real_kd_alpha,
            )
            if ema_relabel:
                _ema_update(teacher_model, model, args.ema_teacher_relabel)
        else:
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch, args.epochs
            )
        val_loss, val_acc, vp, vl = eval_epoch(
            model, val_loader, criterion, device, desc=f"Epoch {epoch:03d}/{args.epochs} [val]"
        )
        scheduler.step()
        val_m = compute_metrics(vl, vp, num_classes)
        print(f"Epoch {epoch:03d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_m['f1']:.4f}")
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            torch.save(model.state_dict(), args.save)
            print(f"  -> saved (val f1 {best_val_f1:.4f})")

    # 테스트
    print("\n=== Test ===")
    model.load_state_dict(torch.load(args.save, map_location=device))
    _, _, preds, labels = eval_epoch(model, test_loader, criterion, device, desc="[test]")
    test_m = compute_metrics(labels, preds, num_classes)
    print_metrics(test_m, title="Test Metrics")
    save_metrics_json(test_m, args.metrics_json)
    save_confusion_matrix(labels, preds, num_classes, args.cm_png)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="RESULT1.csv")
    p.add_argument("--synth", required=True, help="diffusion 생성 npz")
    p.add_argument("--mix_mode", default="aug", choices=["only", "aug"])
    p.add_argument("--teacher_relabel", default=None,
                   help="경로 B(SDFT relabel): real-data teacher .pt 경로. 지정 시 synth는 "
                        "hard label 대신 teacher soft-label(KL)로 학습. mix_mode=aug 전용.")
    p.add_argument("--ema_teacher_relabel", type=float, default=0.0,
                   help=">0이면 frozen teacher 대신 student EMA(decay)를 relabel teacher로 사용(자기증류).")
    p.add_argument("--init_from", default=None,
                   help="student 초기화 checkpoint. teacher_relabel과 같이 쓰면 self-distillation fine-tune.")
    p.add_argument("--relabel_temp", type=float, default=2.0,
                   help="relabel KL temperature")
    p.add_argument("--real_kd_alpha", type=float, default=1.0,
                   help="real sample loss의 CE 비중. 1.0=기존 CE만, 0.5=CE와 teacher KL 혼합.")
    p.add_argument("--relabel_conf_min", type=float, default=0.0,
                   help="합성 샘플 teacher confidence 최소값. 낮으면 synthetic KL에서 제외.")
    p.add_argument("--relabel_conf_weight", default="none",
                   choices=["none", "max", "entropy"],
                   help="합성 KL sample weight. max=teacher max prob, entropy=1-normalized entropy.")
    p.add_argument("--relabel_class_weights_json", default=None,
                   help='합성 KL class weight JSON. 예: {"7":1.5,"9":2.0}')
    p.add_argument("--ce_weight_real_only", action="store_true",
                   help="CE class weight를 real train label로만 계산. teacher_relabel에서 권장.")

    p.add_argument("--samples_per_label", type=int, default=600)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--split_mode", default="temporal", choices=["random","temporal"])
    p.add_argument("--trim_head", type=int, default=10)
    p.add_argument("--trim_tail", type=int, default=30)
    p.add_argument("--merge11", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--arch", default="hybrid_cnn_bilstm",
                   choices=["cnn_bilstm","resnet1d","patchtst",
                            "cross_patchtst","looped_patchtst",
                            "hybrid_cnn_bilstm","hybrid_resnet1d","hybrid_patchtst"])
    # Looped PatchTST 전용
    p.add_argument("--loop_T", type=int, default=3)
    p.add_argument("--loop_inject_stab", action="store_true")
    p.add_argument("--recur_mode", default="update", choices=["update", "prenorm", "gated"])
    p.add_argument("--gate_init", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sample_rate_hz", type=float, default=10.0)
    p.add_argument("--patch_len", type=int, default=5)
    p.add_argument("--patch_stride", type=int, default=5)
    p.add_argument("--patch_pool", default="flatten", choices=["flatten", "mean", "cls"],
                   help="PatchTST pooling. patchtst/hybrid_patchtst는 flatten|mean, "
                        "cross_patchtst는 cls|mean")
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=128)

    p.add_argument("--teacher_arch", default=None,
                   choices=["cnn_bilstm","resnet1d","patchtst",
                            "cross_patchtst",
                            "hybrid_cnn_bilstm","hybrid_resnet1d","hybrid_patchtst"],
                   help="teacher_relabel용 teacher architecture. 미지정 시 student arch와 동일.")
    p.add_argument("--teacher_window", type=int, default=None)
    p.add_argument("--teacher_sample_rate_hz", type=float, default=None)
    p.add_argument("--teacher_patch_len", type=int, default=None)
    p.add_argument("--teacher_patch_stride", type=int, default=None)
    p.add_argument("--teacher_patch_pool", default=None, choices=["flatten", "mean", "cls"])
    p.add_argument("--teacher_d_model", type=int, default=None)
    p.add_argument("--teacher_n_heads", type=int, default=None)
    p.add_argument("--teacher_e_layers", type=int, default=None)
    p.add_argument("--teacher_d_ff", type=int, default=None)

    p.add_argument("--save", default="best_aug.pt")
    p.add_argument("--metrics_json", default="results_json/test_metrics_aug.json")
    p.add_argument("--cm_png", default="confusion_matrices/cm_aug.png")

    args = p.parse_args()
    main(args)
