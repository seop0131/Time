import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    make_splits,
    make_splits_first_per_label,
    make_splits_random,
    make_splits_random_two_csv,
    make_splits_two_csv,
)
from noise_augment import TimeSeriesNoise, RealEnvNoise, DomainRandomNoise
from model import build_model
from metrics import compute_metrics, print_metrics, save_confusion_matrix, save_metrics_json


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_torch_load(path, map_location):
    """torch>=2.6(weights_only 기본 True)과 torch<1.13(weights_only 미지원) 모두 처리."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def train_epoch(model, loader, optimizer, criterion, device, epoch, epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{epochs} [train]", leave=False)
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
        pbar.set_postfix(loss=f"{total_loss / total:.4f}", acc=f"{correct / total:.4f}")
    return total_loss / total, correct / total


def train_epoch_multiexit_sd(
    model, loader, optimizer, criterion, device, epoch, epochs,
    sd_lambda, temperature, kl_weight=1.0,
    ema_teacher=None, ema_decay=0.0, aux_ce=0.0,
):
    """
    Phase 0 — multi-exit self-distillation (LOOPED_SDFT_DESIGN.md §5).
    looped backbone 없이, hpatchfix의 각 encoder layer exit logit z_t에 대해
    deep teacher가 얕은 exit(student)를 가르친다.

      L = CE(z_T, y)
        + aux_ce · Σ_{t<T} CE(z_t, y)                                   # (옵션) 중간 exit deep supervision
        + sd_lambda · kl_weight · Σ_{t<T} T² · KL( sg[p_T(teacher)] || p_T(z_t) )

    teacher 신호:
      - ema_teacher=None : 최종 exit z_T(stop-grad) 가 teacher (기본).
      - ema_teacher!=None: 별도 EMA 모델의 최종 exit 이 teacher. 매 스텝 EMA 갱신.
    """
    model.train()
    if ema_teacher is not None:
        ema_teacher.eval()
    total_loss = total_ce = total_kl = 0.0
    correct = total = 0
    kl_div = nn.KLDivLoss(reduction="batchmean")
    tag = "mexit_ema" if ema_teacher is not None else "mexit_sd"
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{epochs} [{tag}]", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        exits = model.forward_multi_exit(x)   # [z_0, ..., z_T]
        z_final = exits[-1]

        ce = criterion(z_final, y)
        if aux_ce > 0:
            for z_t in exits[:-1]:
                ce = ce + aux_ce * criterion(z_t, y)

        # teacher 분포 (stop-grad)
        with torch.no_grad():
            if ema_teacher is not None:
                t_logits = ema_teacher.forward_multi_exit(x)[-1]
            else:
                t_logits = z_final
            p_teacher = torch.softmax(t_logits / temperature, dim=-1)
        kl = 0.0
        for z_t in exits[:-1]:
            log_p_student = torch.log_softmax(z_t / temperature, dim=-1)
            kl = kl + kl_div(log_p_student, p_teacher) * (temperature ** 2)

        loss = ce + sd_lambda * kl_weight * kl
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if ema_teacher is not None and ema_decay > 0:
            _ema_update(ema_teacher, model, ema_decay)

        bs = len(y)
        total_loss += loss.item() * bs
        total_ce += ce.item() * bs
        total_kl += float(kl) * bs
        correct += (z_final.argmax(1) == y).sum().item()
        total += bs
        pbar.set_postfix(loss=f"{total_loss / total:.4f}", ce=f"{total_ce / total:.4f}",
                         kl=f"{total_kl / total:.4f}", acc=f"{correct / total:.4f}")
    return total_loss / total, correct / total


def gkd_divergence(logits_student, logits_teacher, temperature, beta):
    """
    Generalized JSD(β) divergence (GKD, arXiv 2306.13649). T² 스케일 포함.

      M = β·p_T + (1-β)·p_S
      JSD_β = β·KL(p_T || M) + (1-β)·KL(p_S || M)

    β=1 → forward-KL(p_T||p_S, mode-covering, 기존 KD 동작)
    β=0 → reverse-KL(p_S||p_T, mode-seeking)
    β=0.5 → 대칭 JSD
    p_T는 no_grad(teacher), gradient는 student로만 흐른다.
    """
    p_s = torch.softmax(logits_student / temperature, dim=-1)
    with torch.no_grad():
        p_t = torch.softmax(logits_teacher / temperature, dim=-1)
    if beta >= 1.0:
        # forward-KL: KL(p_T || p_S) = Σ p_T (log p_T - log p_S)
        log_ps = torch.log_softmax(logits_student / temperature, dim=-1)
        div = (p_t * (torch.log(p_t + 1e-8) - log_ps)).sum(-1).mean()
    elif beta <= 0.0:
        # reverse-KL: KL(p_S || p_T)
        log_ps = torch.log_softmax(logits_student / temperature, dim=-1)
        div = (p_s * (log_ps - torch.log(p_t + 1e-8))).sum(-1).mean()
    else:
        m = beta * p_t + (1.0 - beta) * p_s
        logm = torch.log(m + 1e-8)
        kl_tm = (p_t * (torch.log(p_t + 1e-8) - logm)).sum(-1).mean()
        kl_sm = (p_s * (torch.log(p_s + 1e-8) - logm)).sum(-1).mean()
        div = beta * kl_tm + (1.0 - beta) * kl_sm
    return div * (temperature ** 2)


def train_epoch_kd(
    student, teacher, loader, optimizer, criterion, device, epoch, epochs,
    noise_transform, alpha, temperature, noisy_student=True, beta=1.0, lam=1.0,
):
    """
    Generalized Knowledge Distillation (GKD, OPD 일반형) — 분류 버전.
      - Teacher: freeze, no grad, eval
      - Loss = α·CE(student, y)
               + (1-α)·[ (1-λ)·Div(clean) + λ·Div(aug) ]
      - Div = JSD(β) (gkd_divergence). β=1 forward-KL(기존), β=0 reverse-KL.
      - λ = on-policy 비율. λ=1이면 augmented(student가 보는 분포)에서만 채점(기존 동작),
            λ<1이면 clean 입력 채점도 섞음(off-policy 보강).

    noisy_student=True 시 aug 입력은 noise_transform 적용. False면 aug=clean이라
    λ 무관하게 clean만.
    """
    student.train()
    teacher.eval()
    total_loss = total_ce = total_kl = 0.0
    correct = total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{epochs} [gkd]", leave=False)

    for x_clean, y in pbar:
        x_clean = x_clean.to(device)
        y = y.to(device)

        has_aug = noise_transform is not None and noisy_student
        if has_aug:
            x_aug = torch.stack(
                [noise_transform(x_clean[i].cpu()) for i in range(x_clean.size(0))]
            ).to(device)
        else:
            x_aug = x_clean

        with torch.no_grad():
            logits_teacher = teacher(x_clean)  # teacher는 항상 clean 입력 채점

        optimizer.zero_grad()
        # CE는 student가 실제 학습하는 입력(aug 우선)에서
        logits_student_aug = student(x_aug)
        ce = criterion(logits_student_aug, y)

        # GKD divergence: λ로 clean/aug 항 가중
        div = 0.0
        if lam < 1.0:
            logits_student_clean = student(x_clean) if has_aug else logits_student_aug
            div = div + (1.0 - lam) * gkd_divergence(
                logits_student_clean, logits_teacher, temperature, beta)
        if lam > 0.0:
            div = div + lam * gkd_divergence(
                logits_student_aug, logits_teacher, temperature, beta)

        loss = alpha * ce + (1.0 - alpha) * div
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        bs = len(y)
        total_loss += loss.item() * bs
        total_ce += ce.item() * bs
        total_kl += float(div) * bs
        correct += (logits_student_aug.argmax(1) == y).sum().item()
        total += bs
        pbar.set_postfix(
            loss=f"{total_loss / total:.4f}",
            ce=f"{total_ce / total:.4f}",
            div=f"{total_kl / total:.4f}",
            acc=f"{correct / total:.4f}",
        )
    return total_loss / total, correct / total


@torch.no_grad()
def _ema_update(teacher, student, decay):
    """teacher = decay·teacher + (1-decay)·student (weight + buffer)."""
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.mul_(decay).add_(s.detach(), alpha=1.0 - decay)
    for t, s in zip(teacher.buffers(), student.buffers()):
        t.copy_(s)


def train_epoch_sdft(
    model, loader, optimizer, criterion, device, epoch, epochs,
    noise_transform, alpha, temperature, teacher=None, ema_decay=0.0, kl_weight=1.0,
):
    """
    Self-Distillation Fine-Tuning step.

      - Student forward: noise-augmented input, grad
      - Teacher forward: clean input, no_grad
      - Loss = α·CE(student, y) + (1-α)·kl_weight·T²·KL(student^T || teacher^T)

    teacher=None  : 기존(naive) SDFT — 같은 모델이 자기 자신을 teacher로 사용.
                    하지만 model.eval()로 두어 BN/dropout이 안정된 타깃을 주도록 한다.
    teacher!=None : EMA teacher(권장). 매 스텝 EMA로 갱신. 안정적 타깃 → collapse 방지.

    kl_weight: KL warmup용 (0→1 ramp). 초기에 자기증류 잡음을 줄인다.
    loader는 clean batch를 줘야 함. noise_transform이 윈도우별 augmentation.
    """
    model.train()
    if teacher is not None:
        teacher.eval()
    total_loss = total_ce = total_kl = 0.0
    correct = total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{epochs} [sdft]", leave=False)
    kl_div = nn.KLDivLoss(reduction="batchmean")

    for x_clean, y in pbar:
        x_clean = x_clean.to(device)
        y = y.to(device)

        # 윈도우별 노이즈 적용
        x_noisy = torch.stack([noise_transform(x_clean[i].cpu()) for i in range(x_clean.size(0))]).to(device)

        # Teacher: clean, no grad. EMA teacher가 있으면 그것을, 없으면 자기 자신(eval).
        with torch.no_grad():
            if teacher is not None:
                logits_teacher = teacher(x_clean)
            else:
                model.eval()
                logits_teacher = model(x_clean)
                model.train()
            p_teacher = torch.softmax(logits_teacher / temperature, dim=-1)

        # Student: noisy, grad O
        optimizer.zero_grad()
        logits_student = model(x_noisy)
        log_p_student = torch.log_softmax(logits_student / temperature, dim=-1)

        ce = criterion(logits_student, y)
        kl = kl_div(log_p_student, p_teacher) * (temperature ** 2)
        loss = alpha * ce + (1.0 - alpha) * kl_weight * kl

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # EMA teacher 갱신 (student 업데이트 직후)
        if teacher is not None and ema_decay > 0:
            _ema_update(teacher, model, ema_decay)

        bs = len(y)
        total_loss += loss.item() * bs
        total_ce += ce.item() * bs
        total_kl += kl.item() * bs
        correct += (logits_student.argmax(1) == y).sum().item()
        total += bs
        pbar.set_postfix(
            loss=f"{total_loss / total:.4f}",
            ce=f"{total_ce / total:.4f}",
            kl=f"{total_kl / total:.4f}",
            acc=f"{correct / total:.4f}",
        )
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, desc="eval"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 데이터 준비
    if args.first_per_label:
        # 각 label의 첫 N행만 추출 후 split (random / temporal 선택)
        train_ds, val_ds, test_ds = make_splits_first_per_label(
            args.csv,
            samples_per_label=args.samples_per_label,
            window=args.window,
            stride=args.stride,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            binary=args.binary,
            seed=args.seed,
            split_mode=args.split_mode,
            trim_head=args.trim_head,
            trim_tail=args.trim_tail,
            merge11=args.merge11,
        )
    elif args.random_split and args.test_csv:
        # train_csv를 random으로 train/val 분할, test_csv 전체를 test로
        train_ds, val_ds, test_ds = make_splits_random_two_csv(
            train_csv=args.csv,
            test_csv=args.test_csv,
            window=args.window,
            stride=args.stride,
            val_ratio=args.val_ratio,
            binary=args.binary,
            seed=args.seed,
        )
    elif args.random_split:
        # [진단용] 단일 csv를 윈도우 단위 완전 random split — train/val/test 모두 누수 가능
        train_ds, val_ds, test_ds = make_splits_random(
            args.csv,
            window=args.window,
            stride=args.stride,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            binary=args.binary,
            seed=args.seed,
        )
    elif args.test_csv:
        # train_csv → train+val, test_csv → test 전체
        train_ds, val_ds, test_ds = make_splits_two_csv(
            train_csv=args.csv,
            test_csv=args.test_csv,
            window=args.window,
            stride=args.stride,
            val_ratio=args.val_ratio,
            binary=args.binary,
        )
    else:
        # 단일 csv를 시간 순서 기준 train/val/test로 분할
        train_ds, val_ds, test_ds = make_splits(
            args.csv,
            window=args.window,
            stride=args.stride,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            binary=args.binary,
        )

    # online noise augmentation — train_ds에만 적용 (val/test는 깨끗하게 평가)
    sdft_noise = None
    if args.noise:
        if args.noise_mode == "domain":
            noise = DomainRandomNoise(
                base_noise_std=args.dom_base,
                snr_range=(args.dom_snr_min, args.dom_snr_max),
                ar1_phi_range=(args.dom_ar1_min, args.dom_ar1_max),
                cross_channel_ratio=args.dom_cross,
                smooth_prob=args.dom_smooth_p,
                sat_prob=args.dom_sat_p,
                bias_drift_std=args.dom_bias_drift,
                quantize_prob=args.dom_quant_p,
                p_apply=args.noise_p,
                enabled=True,
            )
        elif args.noise_mode == "real":
            noise = RealEnvNoise(
                gaussian_std=args.noise_gaussian,
                amp_jitter=args.noise_amp,
                bias_std=args.noise_bias,
                spike_prob=args.noise_spike_prob,
                spike_std=args.noise_spike_std,
                drift_amp=args.noise_drift,
                pink_std=args.noise_pink,
                hum_amp=args.noise_hum,
                hum_freq_hz=args.noise_hum_freq,
                boundary_bias_prob=args.noise_bb_prob,
                boundary_bias_max=args.noise_bb_max,
                mask_prob=args.noise_mask_prob,
                mask_ratio=args.noise_mask_ratio,
                p_apply=args.noise_p,
                enabled=True,
            )
        else:
            noise = TimeSeriesNoise(
                gaussian_std=args.noise_gaussian,
                amp_jitter=args.noise_amp,
                bias_std=args.noise_bias,
                spike_prob=args.noise_spike_prob,
                spike_std=args.noise_spike_std,
                p_apply=args.noise_p,
                enabled=True,
            )
        # KD with pretrained teacher 또는 SDFT 모두 batch별 noise 적용 필요 → transform 비활성
        if args.sdft or args.kd_teacher:
            sdft_noise = noise
            train_ds.transform = None
            mode = "SDFT" if args.sdft else "KD"
            print(f"{mode} mode — clean teacher / noisy student ({noise})")
        else:
            train_ds.transform = noise
            sdft_noise = None
            print(f"Noise augmentation: {noise}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    # 클래스 불균형 보정
    label_counts = np.bincount(train_ds.targets)
    class_weights = 1.0 / (label_counts + 1e-6)
    class_weights = torch.tensor(class_weights / class_weights.sum() * len(label_counts), dtype=torch.float32).to(device)

    # 모델
    num_classes = int(train_ds.targets.max()) + 1
    model_kwargs = {}
    patch_pool = args.patch_pool
    if args.arch == "cross_patchtst" and patch_pool == "flatten":
        patch_pool = "cls"
    if args.arch in {"patchtst", "cross_patchtst"}:
        model_kwargs.update(
            seq_len=args.window,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_ff=args.d_ff,
            pool=patch_pool,
        )
    elif args.arch == "looped_patchtst":
        pool = "mean" if patch_pool == "cls" else patch_pool
        model_kwargs.update(
            seq_len=args.window,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            loop_T=args.loop_T,
            inject_stab=args.loop_inject_stab,
            recur_mode=args.recur_mode,
            gate_init=args.gate_init,
            pool=pool,
        )
    elif args.arch.startswith("hybrid_"):
        model_kwargs["window"] = args.window
        model_kwargs["sample_rate_hz"] = args.sample_rate_hz
        model_kwargs["freq_arch"] = args.freq_arch
        if args.freq_arch == "patchtst":
            # freq 축(W//2+1)은 매우 짧으므로 overlapping patch 사용
            model_kwargs["freq_patchtst_kwargs"] = dict(
                patch_len=2,
                stride=1,
                d_model=args.d_model,
                n_heads=args.n_heads,
                e_layers=args.e_layers,
                d_ff=args.d_ff,
                pool="mean",
            )
        if args.arch == "hybrid_patchtst":
            model_kwargs["patchtst_kwargs"] = dict(
                seq_len=args.window,
                patch_len=args.patch_len,
                stride=args.patch_stride,
                d_model=args.d_model,
                n_heads=args.n_heads,
                e_layers=args.e_layers,
                d_ff=args.d_ff,
                pool=patch_pool,
            )
    model = build_model(num_classes=num_classes, arch=args.arch, **model_kwargs).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # SDFT는 본래 fine-tuning 단계 — 수렴된 baseline에서 출발하면 collapse가 크게 준다.
    if args.init_from:
        sd = _safe_torch_load(args.init_from, map_location=device)
        model.load_state_dict(sd)
        print(f"Initialized model from {args.init_from} (fine-tune mode)")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # SDFT 사용 시 noise가 반드시 있어야 함
    use_sdft = args.sdft and args.noise
    if args.sdft and not args.noise:
        print("WARNING: --sdft requires --noise. SDFT disabled.")

    # EMA teacher (권장): 안정적 자기증류 타깃 → naive SDFT의 confirmation-bias collapse 방지
    ema_teacher = None
    if use_sdft and args.sdft_ema > 0:
        ema_teacher = build_model(num_classes=num_classes, arch=args.arch, **model_kwargs).to(device)
        ema_teacher.load_state_dict(model.state_dict())
        for p in ema_teacher.parameters():
            p.requires_grad_(False)
        ema_teacher.eval()
    if use_sdft:
        print(f"SDFT enabled: alpha={args.sdft_alpha}, temperature={args.sdft_temperature}, "
              f"ema_decay={args.sdft_ema}, kl_warmup={args.sdft_kl_warmup}ep, "
              f"teacher={'EMA' if ema_teacher is not None else 'self(eval)'}")

    # multi-exit SD용 EMA teacher (옵션)
    mexit_ema_teacher = None
    if args.multiexit_sd and args.multiexit_ema > 0:
        mexit_ema_teacher = build_model(num_classes=num_classes, arch=args.arch, **model_kwargs).to(device)
        mexit_ema_teacher.load_state_dict(model.state_dict())
        for p in mexit_ema_teacher.parameters():
            p.requires_grad_(False)
        mexit_ema_teacher.eval()
    if args.multiexit_sd:
        print(f"multi-exit SD: lambda={args.multiexit_lambda}, temp={args.multiexit_temp}, "
              f"warmup={args.multiexit_warmup}ep, ema={args.multiexit_ema}, "
              f"aux_ce={args.multiexit_aux_ce}, "
              f"teacher={'EMA' if mexit_ema_teacher is not None else 'final-exit(sg)'}")

    # KD with pretrained teacher
    teacher_model = None
    if args.kd_teacher:
        print(f"Loading KD teacher from {args.kd_teacher}")
        teacher_model = build_model(num_classes=num_classes, arch=args.arch, **model_kwargs).to(device)
        teacher_sd = _safe_torch_load(args.kd_teacher, map_location=device)
        try:
            teacher_model.load_state_dict(teacher_sd)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                # 보통 teacher는 merge11(11class)인데 student split이 12class로 잡힌 경우.
                raise SystemExit(
                    f"[KD teacher mismatch] teacher='{args.kd_teacher}' 와 현재 모델의 "
                    f"클래스 수가 다릅니다 (현재 num_classes={num_classes}). "
                    f"teacher 학습 시 사용한 --merge11 / --samples_per_label 설정을 "
                    f"student 쪽에도 동일하게 주세요.\n  원본 오류: {e}"
                )
            raise
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        teacher_model.eval()
        print(f"KD enabled: alpha={args.sdft_alpha}, T={args.sdft_temperature}, "
              f"noisy_student={args.noise is not False and sdft_noise is not None}")

    best_val_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        if teacher_model is not None:
            # Pretrained teacher KD
            train_loss, train_acc = train_epoch_kd(
                model, teacher_model, train_loader, optimizer, criterion, device, epoch, args.epochs,
                noise_transform=sdft_noise,
                alpha=args.sdft_alpha, temperature=args.sdft_temperature,
                noisy_student=(sdft_noise is not None),
                beta=args.gkd_beta, lam=args.gkd_lambda,
            )
        elif args.multiexit_sd:
            # Phase 0 multi-exit self-distillation (LOOPED_SDFT_DESIGN.md)
            if args.multiexit_warmup > 0:
                kl_w = min(1.0, epoch / float(args.multiexit_warmup))
            else:
                kl_w = 1.0
            train_loss, train_acc = train_epoch_multiexit_sd(
                model, train_loader, optimizer, criterion, device, epoch, args.epochs,
                sd_lambda=args.multiexit_lambda, temperature=args.multiexit_temp,
                kl_weight=kl_w,
                ema_teacher=mexit_ema_teacher, ema_decay=args.multiexit_ema,
                aux_ce=args.multiexit_aux_ce,
            )
        elif use_sdft:
            # KL warmup: 처음 sdft_kl_warmup epoch 동안 KL 가중치를 0→1로 선형 증가
            if args.sdft_kl_warmup > 0:
                kl_w = min(1.0, epoch / float(args.sdft_kl_warmup))
            else:
                kl_w = 1.0
            train_loss, train_acc = train_epoch_sdft(
                model, train_loader, optimizer, criterion, device, epoch, args.epochs,
                noise_transform=sdft_noise,
                alpha=args.sdft_alpha, temperature=args.sdft_temperature,
                teacher=ema_teacher, ema_decay=args.sdft_ema, kl_weight=kl_w,
            )
        else:
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch, args.epochs
            )
        val_loss, val_acc, val_preds, val_labels = eval_epoch(
            model, val_loader, criterion, device, desc=f"Epoch {epoch:03d}/{args.epochs} [val]"
        )
        scheduler.step()

        val_m = compute_metrics(val_labels, val_preds, num_classes)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_m['f1']:.4f}"
        )

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            torch.save(model.state_dict(), args.save)
            print(f"  -> saved (val f1 {best_val_f1:.4f})")

    # 테스트
    print("\n=== Test ===")
    model.load_state_dict(torch.load(args.save, map_location=device))
    _, test_acc, preds, labels = eval_epoch(model, test_loader, criterion, device, desc="[test]")

    test_m = compute_metrics(labels, preds, num_classes)
    print_metrics(test_m, title="Test Metrics")
    save_metrics_json(test_m, args.metrics_json)
    save_confusion_matrix(labels, preds, num_classes, args.cm_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="RESULT1.csv", help="학습용 CSV (test_csv 미지정 시 train/val/test 분할)")
    parser.add_argument("--test_csv", default=None, help="테스트용 CSV. 지정 시 csv는 train+val, test_csv 전체를 test로 사용")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--save", default="best_model.pt")
    parser.add_argument("--metrics_json", default="results_json/test_metrics.json")
    parser.add_argument("--cm_png", default="confusion_matrices/confusion_matrix.png")
    parser.add_argument("--binary", action="store_true", help="저부하(0,1,7-11)/고부하(2-6) 이진분류")
    parser.add_argument("--merge11", action="store_true",
                        help="label 11(동작 무부하)을 label 0(휴식)으로 통합. 11-class 분류.")
    parser.add_argument("--random_split", action="store_true",
                        help="[진단용] 윈도우 단위 완전 random split. 누수 가능 - 일반화 평가에는 부적합")
    parser.add_argument("--seed", type=int, default=42, help="random_split 사용 시 셔플 시드")
    parser.add_argument("--first_per_label", action="store_true",
                        help="각 label의 첫 N행만 추출해 윈도우 random split (반복 누수 차단)")
    parser.add_argument("--samples_per_label", type=int, default=600,
                        help="label당 추출 행 수 (0.1초 간격 기준 600=60초)")
    parser.add_argument("--split_mode", default="random", choices=["random", "temporal"],
                        help="first_per_label에서 split 방식: random(윈도우 셔플) / temporal(시간순)")
    parser.add_argument("--trim_head", type=int, default=0,
                        help="각 클래스 구간의 앞쪽 행 수 제거 (0.1초 간격 기준 10=1초)")
    parser.add_argument("--trim_tail", type=int, default=0,
                        help="각 클래스 구간의 뒤쪽 행 수 제거 (0.1초 간격 기준 30=3초)")
    # Online noise augmentation (train에만 적용)
    parser.add_argument("--noise", action="store_true",
                        help="train 윈도우에 online noise augmentation 적용")
    parser.add_argument("--noise_gaussian", type=float, default=0.05,
                        help="Gaussian 노이즈 σ (정규화된 값 단위)")
    parser.add_argument("--noise_amp", type=float, default=0.05,
                        help="진폭 jitter ±비율 (0.05=±5%)")
    parser.add_argument("--noise_bias", type=float, default=0.05,
                        help="DC bias σ")
    parser.add_argument("--noise_spike_prob", type=float, default=0.01,
                        help="spike 발생 확률")
    parser.add_argument("--noise_spike_std", type=float, default=0.3,
                        help="spike 크기 σ")
    parser.add_argument("--noise_p", type=float, default=1.0,
                        help="윈도우별 노이즈 적용 확률 (1.0=항상)")
    # RealEnvNoise 전용 (--noise_mode real 일 때만 사용)
    parser.add_argument("--noise_mode", default="basic",
                        choices=["basic", "real", "domain"],
                        help="basic: 4종 / real: drift+pink+hum 종합 / domain: AR(1)+cross-ch+SNR 랜덤")
    # DomainRandomNoise 전용
    parser.add_argument("--dom_base", type=float, default=0.1, help="domain base σ")
    parser.add_argument("--dom_snr_min", type=float, default=0.5)
    parser.add_argument("--dom_snr_max", type=float, default=3.0)
    parser.add_argument("--dom_ar1_min", type=float, default=0.5)
    parser.add_argument("--dom_ar1_max", type=float, default=0.95)
    parser.add_argument("--dom_cross", type=float, default=0.5,
                        help="공통 노이즈 비율 (1.0=모두 공통)")
    parser.add_argument("--dom_smooth_p", type=float, default=0.5)
    parser.add_argument("--dom_sat_p", type=float, default=0.02)
    parser.add_argument("--dom_bias_drift", type=float, default=0.3)
    parser.add_argument("--dom_quant_p", type=float, default=0.3)
    # Self-Distillation Fine-Tuning
    parser.add_argument("--sdft", action="store_true",
                        help="Online Self-Distillation. clean teacher / noisy student. --noise 필요")
    parser.add_argument("--sdft_alpha", type=float, default=0.5,
                        help="CE 비중 (0.5=CE와 KL 동등). 1.0=일반 학습, 0.0=KL만")
    parser.add_argument("--sdft_temperature", type=float, default=4.0,
                        help="distillation temperature (보통 2~4)")
    parser.add_argument("--sdft_ema", type=float, default=0.0,
                        help="SDFT EMA teacher decay (0=naive self-teacher, 0.99~0.999 권장). "
                             ">0이면 student의 EMA를 안정적 teacher로 사용해 collapse 방지")
    parser.add_argument("--sdft_kl_warmup", type=int, default=0,
                        help="처음 N epoch 동안 KL 가중치를 0→1로 선형 증가 (초기 자기증류 잡음 완화)")
    parser.add_argument("--init_from", default=None,
                        help="시작 가중치 (.pt). SDFT를 수렴된 baseline에서 fine-tune할 때 사용")
    parser.add_argument("--gkd_beta", type=float, default=1.0,
                        help="GKD divergence 보간 β: 1=forward-KL(기존), 0=reverse-KL(mode-seeking), 0.5=JSD")
    parser.add_argument("--gkd_lambda", type=float, default=1.0,
                        help="GKD on-policy 비율 λ: 1=aug 입력만 채점(기존), 0=clean만, 0.5=혼합")
    parser.add_argument("--sample_rate_hz", type=float, default=10.0,
                        help="입력 신호 샘플링 주파수(Hz). FFT bin → Hz 매핑에 사용. "
                             "본 데이터는 0.1초 간격이므로 기본 10.0")
    parser.add_argument("--kd_teacher", default=None,
                        help="사전 학습된 teacher 가중치 경로 (.pt). 지정 시 KD 모드. "
                             "alpha/temperature는 --sdft_alpha/--sdft_temperature와 공유.")
    parser.add_argument("--noise_drift", type=float, default=0.5,
                        help="drift 진폭 (정규화 단위)")
    parser.add_argument("--noise_pink", type=float, default=0.1,
                        help="pink noise σ")
    parser.add_argument("--noise_hum", type=float, default=0.05,
                        help="60Hz hum 진폭")
    parser.add_argument("--noise_hum_freq", type=float, default=60.0,
                        help="hum 주파수 (Hz)")
    parser.add_argument("--noise_bb_prob", type=float, default=0.3,
                        help="경계 침범 큰 bias 적용 확률")
    parser.add_argument("--noise_bb_max", type=float, default=0.7,
                        help="경계 침범 bias 최대 ± (정규화 단위, z 0.7≈실제 약 1A)")
    parser.add_argument("--noise_mask_prob", type=float, default=0.1,
                        help="마스킹 발생 확률")
    parser.add_argument("--noise_mask_ratio", type=float, default=0.2,
                        help="마스킹 길이 비율")
    parser.add_argument("--arch", default="cnn_bilstm",
                        choices=["cnn_bilstm", "resnet1d", "patchtst",
                                 "cross_patchtst", "looped_patchtst",
                                 "hybrid_cnn_bilstm", "hybrid_resnet1d", "hybrid_patchtst"],
                        help="모델 아키텍처 선택. hybrid_*는 time+freq 결합. "
                             "looped_patchtst는 weight-tied loop backbone")
    # Looped PatchTST 전용 (LOOPED_SDFT_DESIGN.md Phase 1/2)
    parser.add_argument("--loop_T", type=int, default=3, help="loop 반복 횟수 T")
    parser.add_argument("--loop_inject_stab", action="store_true",
                        help="W_inj 음의 대각(-0.1·I) 초기화")
    parser.add_argument("--recur_mode", default="update",
                        choices=["update", "prenorm", "gated"],
                        help="재귀 갱신 방식. update(기존) / prenorm(state pre-norm) / gated(convex)")
    parser.add_argument("--gate_init", type=float, default=0.5, help="gated 모드 초기 α")
    # PatchTST 전용 — 다른 arch에서는 무시
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--patch_pool", default="flatten", choices=["flatten", "mean", "cls"],
                        help="PatchTST pooling. patchtst/hybrid_patchtst는 flatten|mean, "
                             "cross_patchtst는 cls|mean")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=256)
    # hybrid_* frequency branch 인코더 선택
    parser.add_argument("--freq_arch", default="cnn", choices=["cnn", "patchtst"],
                        help="hybrid_* 모델의 frequency branch 인코더. "
                             "cnn(기존) 또는 patchtst(freq 축을 시퀀스로 PatchTST 인코딩)")
    # Phase 0 multi-exit self-distillation (LOOPED_SDFT_DESIGN.md)
    parser.add_argument("--multiexit_sd", action="store_true",
                        help="hybrid_patchtst의 각 encoder layer exit에 deep→shallow self-distill KL 적용")
    parser.add_argument("--multiexit_lambda", type=float, default=1.0,
                        help="multi-exit KL 가중치 λ")
    parser.add_argument("--multiexit_temp", type=float, default=2.0,
                        help="multi-exit KL temperature")
    parser.add_argument("--multiexit_warmup", type=int, default=5,
                        help="multi-exit KL warmup epoch (0→1 ramp)")
    parser.add_argument("--multiexit_ema", type=float, default=0.0,
                        help="multi-exit EMA teacher decay (>0이면 별도 EMA 모델의 최종 exit이 teacher)")
    parser.add_argument("--multiexit_aux_ce", type=float, default=0.0,
                        help="중간 exit auxiliary CE 가중치 (deep supervision)")
    args = parser.parse_args()
    main(args)
