import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# PatchTST 원본 백본을 그대로 쓰기 위해 PatchTST_supervised 경로를 import path에 추가
_PATCHTST_ROOT = os.path.join(os.path.dirname(__file__), "PatchTST", "PatchTST_supervised")
if _PATCHTST_ROOT not in sys.path:
    sys.path.insert(0, _PATCHTST_ROOT)


class BasicBlock1D(nn.Module):
    """ResNet basic block (1D). conv-bn-relu-conv-bn + skip → relu."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 입력 채널/해상도가 바뀌면 skip도 맞춰서 1×1 conv로 투영
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out, inplace=True)


class ResNet1D(nn.Module):
    """
    1D ResNet 시계열 분류 모델.

    Input:  (batch, window, 4)  ← train.py가 넘기는 형식
    Output: (batch, num_classes)

    구조: stem → 4 stage (각 stage에 block_per_stage개의 BasicBlock1D) → GAP → FC.
    좁은 window(10~100 스텝)에서도 동작하도록 stem은 stride 1, 풀링 생략.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 12,
        stem_channels: int = 32,
        stage_channels: tuple[int, ...] = (32, 64, 128, 256),
        blocks_per_stage: int = 2,
        kernel: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        # stem — 첫 conv. window가 짧을 수 있어 stride 1, kernel 7 정도로만 가볍게.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )

        # 4 stage. 첫 stage는 stride 1, 나머지는 stride 2로 점진적 다운샘플.
        stages = []
        in_ch = stem_channels
        for i, out_ch in enumerate(stage_channels):
            stride = 1 if i == 0 else 2
            blocks = [BasicBlock1D(in_ch, out_ch, stride=stride, kernel=kernel, dropout=dropout)]
            for _ in range(blocks_per_stage - 1):
                blocks.append(BasicBlock1D(out_ch, out_ch, stride=1, kernel=kernel, dropout=dropout))
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # GAP → FC
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_ch, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, C) → (B, C, W)
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x).squeeze(-1)  # (B, C)
        return self.classifier(x)


class PatchTSTClassifier(nn.Module):
    """
    원본 PatchTST 백본(Channel Independence + Patch + Transformer)을 그대로 사용하고
    forecasting head 대신 분류 head를 붙인 모델.

    원본 흐름:
        x: (B, W, C) → permute → (B, C, W)
        backbone 내부에서 unfold하여 patch 분할 → Transformer 인코더
        backbone 출력: (B, C, d_model, patch_num)   (원본은 여기서 Flatten_Head로 forecasting)

    여기서는 patch_num 축까지 평균(또는 flatten)해서 (B, C*d_model)로 만들고 FC.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 12,
        seq_len: int = 100,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        fc_dropout: float = 0.2,
        head_dropout: float = 0.0,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        padding_patch: str = "end",
        pool: str = "flatten",  # "flatten" | "mean"
    ):
        super().__init__()
        # 원본 import (sys.path에 PatchTST_supervised 등록 후)
        from layers.PatchTST_backbone import PatchTST_backbone

        # 좁은 window에서 patch_len이 seq_len보다 커지지 않도록 보정
        patch_len = min(patch_len, seq_len)
        stride = min(stride, patch_len)

        # 원본 backbone을 그대로 생성. target_window는 어차피 우리가 head를 떼고 쓰므로
        # 임의 값을 줘도 되지만, Flatten_Head 초기화에 쓰이므로 양수면 충분.
        self.backbone = PatchTST_backbone(
            c_in=in_channels,
            context_window=seq_len,
            target_window=1,                # placeholder, 우리는 backbone.head를 무시
            patch_len=patch_len,
            stride=stride,
            n_layers=e_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            individual=False,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            pretrain_head=False,
            head_type="flatten",
            verbose=False,
        )

        # patch_num 계산 (forward 한 번 트레이싱 없이도 결정 가능)
        # 원본 backbone은 padding_patch='end'일 때 patch_num을 +1 함
        n_patches = (seq_len - patch_len) // stride + 1
        if padding_patch == "end":
            n_patches += 1
        self.n_vars = in_channels
        self.n_patches = n_patches
        self.d_model = d_model
        self.pool = pool

        if pool == "flatten":
            head_in = in_channels * d_model * n_patches
        elif pool == "mean":
            head_in = in_channels * d_model
        else:
            raise ValueError(f"unknown pool: {pool}")

        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(fc_dropout),
            nn.Linear(head_in, num_classes),
        )

    def _backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        원본 backbone의 forward를 head 직전까지만 재현해 (B, C, d_model, patch_num) 반환.
        원본 PatchTST_backbone.forward 코드를 그대로 따라간다.
        """
        z = x  # (B, C, W)
        # RevIN
        if self.backbone.revin:
            z = z.permute(0, 2, 1)
            z = self.backbone.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)
        # padding_patch
        if self.backbone.padding_patch == "end":
            z = self.backbone.padding_patch_layer(z)
        # unfold → (B, C, patch_num, patch_len)
        z = z.unfold(dimension=-1, size=self.backbone.patch_len, step=self.backbone.stride)
        # (B, C, patch_len, patch_num) 형태로 backbone_main이 기대
        z = z.permute(0, 1, 3, 2)
        # backbone_main: TSTiEncoder → (B, C, d_model, patch_num)
        z = self.backbone.backbone(z)
        return z

    def _backbone_features_all_layers(self, x: torch.Tensor) -> list:
        """
        Phase 0(multi-exit SD)용. 각 Transformer encoder layer 출력에서
        (B, C, d_model, patch_num) feature를 뽑아 layer 수만큼 리스트로 반환.
        원본 TSTiEncoder.forward를 따라가되 encoder layer 루프를 펼친다.
        res_attention=False(기본) 가정.
        """
        enc = self.backbone.backbone  # TSTiEncoder
        z = x  # (B, C, W)
        if self.backbone.revin:
            z = z.permute(0, 2, 1)
            z = self.backbone.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)
        if self.backbone.padding_patch == "end":
            z = self.backbone.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.backbone.patch_len, step=self.backbone.stride)
        z = z.permute(0, 1, 3, 2)  # (B, C, patch_len, patch_num) — TSTiEncoder 입력 형태

        # --- TSTiEncoder.forward 재현 (encoder 직전까지) ---
        n_vars = z.shape[1]
        zz = z.permute(0, 1, 3, 2)                       # (B, C, patch_num, patch_len)
        zz = enc.W_P(zz)                                 # (B, C, patch_num, d_model)
        u = torch.reshape(zz, (zz.shape[0] * zz.shape[1], zz.shape[2], zz.shape[3]))
        u = enc.dropout(u + enc.W_pos)                  # (B*C, patch_num, d_model)

        outs = []
        output = u
        scores = None
        res_attn = enc.encoder.res_attention
        for mod in enc.encoder.layers:                   # 각 layer 출력 수집
            if res_attn:
                output, scores = mod(output, prev=scores)
            else:
                output = mod(output)
            zt = torch.reshape(output, (-1, n_vars, output.shape[-2], output.shape[-1]))
            zt = zt.permute(0, 1, 3, 2)                  # (B, C, d_model, patch_num)
            outs.append(zt)
        return outs

    def _pool_flatten(self, feat: torch.Tensor) -> torch.Tensor:
        """(B, C, d_model, patch_num) → (B, head_in) 기존 pool 규칙 적용."""
        if self.pool == "flatten":
            return feat.flatten(1)
        return feat.mean(dim=-1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, C) → (B, C, W)
        x = x.permute(0, 2, 1)
        feat = self._backbone_features(x)        # (B, C, d_model, patch_num)
        feat = self._pool_flatten(feat)
        return self.classifier(feat)


class CrossChannelPatchTSTClassifier(nn.Module):
    """
    Utica-inspired short-window transformer for multivariate sensor classification.

    Unlike vanilla PatchTST, this treats each (channel, patch) pair as a token so
    attention can mix channels directly. Each token contains raw patch values,
    first differences, and patch-level mean/std statistics.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 12,
        seq_len: int = 10,
        patch_len: int = 3,
        stride: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.2,
        pool: str = "cls",  # "cls" | "mean"
    ):
        super().__init__()
        if pool not in {"cls", "mean"}:
            raise ValueError(f"cross-channel patch model supports pool='cls' or 'mean', got {pool}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model({d_model}) must be divisible by n_heads({n_heads})")

        patch_len = min(patch_len, seq_len)
        stride = min(stride, patch_len)
        n_patches = (seq_len - patch_len) // stride + 1
        if n_patches < 1:
            raise ValueError(
                f"invalid patch setup: seq_len={seq_len}, patch_len={patch_len}, stride={stride}"
            )

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = n_patches
        self.pool = pool

        token_dim = patch_len * 2 + 2  # raw patch + first diff + mean + std
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.channel_embed = nn.Parameter(torch.zeros(1, in_channels, 1, d_model))
        self.patch_embed = nn.Parameter(torch.zeros(1, 1, n_patches, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.trunc_normal_(self.patch_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, C) -> (B, C, W)
        xt = x.permute(0, 2, 1)
        patches = xt.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # patches: (B, C, N, L)
        diff = torch.zeros_like(patches)
        diff[..., :-1] = patches[..., 1:] - patches[..., :-1]
        mean = patches.mean(dim=-1, keepdim=True)
        std = patches.std(dim=-1, keepdim=True)
        tokens = torch.cat([patches, diff, mean, std], dim=-1)

        tokens = self.token_proj(tokens)
        tokens = tokens + self.channel_embed + self.patch_embed
        B = tokens.size(0)
        tokens = tokens.reshape(B, self.in_channels * self.n_patches, -1)
        cls = self.cls_token.expand(B, -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))

        if self.pool == "cls":
            feat = encoded[:, 0]
        else:
            feat = encoded[:, 1:].mean(dim=1)
        return self.classifier(feat)


class LoopedPatchTSTClassifier(nn.Module):
    """
    Looped (weight-tied recurrent) PatchTST — LoopViT / Parcae식 (LOOPED_SDFT_DESIGN.md §3).

    overlap patch 임베딩 후, **하나의** TSTEncoderLayer 블록 f_θ를 T번 재귀 적용한다.
    매 step 입력 임베딩 e를 injection으로 재주입한다.

    update 모드 (recur_mode):
      - "update"  : h = h + LN( f(h) + W_inj·e )                 # 기존(불안정 — update에 norm)
      - "prenorm" : h = h + f(LN(h)) + W_inj·e                   # state에 pre-norm 재귀 (Parcae식)
      - "gated"   : h = (1-α)·h + α·( f(LN(h)) + W_inj·e )       # convex update, α∈(0,1) 학습
                    → state가 convex combination이라 ‖h‖ 발산 억제(가장 강한 안정화)

    - inject_stab: W_inj를 **순수 음의 대각(-c·I)**으로 초기화(spectral_norm 미사용 →
      eye-init 충돌 제거). 음의 대각은 임베딩을 약하게 빼주어 residual 누적을 상쇄.
    - depth(반복 T)를 파라미터와 분리(weight-tied) → looped architecture.
    - forward_all_steps(..., return_norms=True)로 per-step ‖h_t‖ 로깅 가능.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 11,
        seq_len: int = 10,
        patch_len: int = 3,
        stride: int = 1,
        d_model: int = 64,
        n_heads: int = 8,
        d_ff: int = 128,
        loop_T: int = 3,
        dropout: float = 0.2,
        pool: str = "mean",          # "mean" | "flatten"
        inject_stab: bool = False,   # W_inj 음의 대각 초기화
        recur_mode: str = "update",  # "update" | "prenorm" | "gated"
        gate_init: float = 0.5,      # gated 모드 초기 α
        return_feat_dim: bool = False,
    ):
        super().__init__()
        from layers.PatchTST_backbone import TSTEncoderLayer
        from layers.PatchTST_layers import positional_encoding

        if recur_mode not in {"update", "prenorm", "gated"}:
            raise ValueError(f"unknown recur_mode: {recur_mode}")

        patch_len = min(patch_len, seq_len)
        stride = min(stride, patch_len)
        n_patches = (seq_len - patch_len) // stride + 1
        # padding_patch='end' 효과를 직접 구현해 토큰 1개 추가(time branch와 동일 규칙)
        self.pad_patch = True
        if self.pad_patch:
            n_patches += 1

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = n_patches
        self.d_model = d_model
        self.loop_T = loop_T
        self.pool = pool
        self.recur_mode = recur_mode

        # patch embedding (channel-independent)
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = positional_encoding("zeros", True, n_patches, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # weight-tied 블록 1개 — T번 재귀
        self.block = TSTEncoderLayer(
            n_patches, d_model, n_heads=n_heads, d_ff=d_ff,
            norm="BatchNorm", attn_dropout=0.0, dropout=dropout,
            activation="gelu", res_attention=False, pre_norm=False,
        )

        # injection: 임베딩 e를 매 step 재주입
        self.W_inj = nn.Linear(d_model, d_model, bias=False)
        if inject_stab:
            # 순수 음의 대각 초기화 (spectral_norm 미사용 → eye-init 충돌 없음)
            with torch.no_grad():
                self.W_inj.weight.copy_(-0.1 * torch.eye(d_model))
        # 재귀 안정화용 norm
        self.state_norm = nn.LayerNorm(d_model)   # prenorm/gated: state에 적용
        self.inject_norm = nn.LayerNorm(d_model)  # update(기존): update에 적용
        # gated: convex 계수 α = sigmoid(gate_logit)
        if recur_mode == "gated":
            import math
            g0 = max(min(gate_init, 1 - 1e-4), 1e-4)
            self.gate_logit = nn.Parameter(torch.tensor(math.log(g0 / (1 - g0))))

        head_in = in_channels * d_model * (n_patches if pool == "flatten" else 1)
        self.head_in = head_in
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(dropout),
            nn.Linear(head_in, num_classes),
        )

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, C) → patch embed → (B*C, n_patches, d_model)
        xt = x.permute(0, 2, 1)  # (B, C, W)
        if self.pad_patch:
            xt = F.pad(xt, (0, self.stride), mode="replicate")
        z = xt.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (B, C, n_patches, patch_len)
        z = self.W_P(z)                                                     # (B, C, n_patches, d_model)
        B, C, N, D = z.shape
        u = z.reshape(B * C, N, D)
        u = self.emb_dropout(u + self.W_pos)
        return u, B, C

    def _pool_head(self, h, B, C):
        # h: (B*C, n_patches, d_model) → logit
        N, D = h.shape[1], h.shape[2]
        h = h.reshape(B, C, N, D)
        if self.pool == "flatten":
            feat = h.reshape(B, -1)
        else:  # mean over patches
            feat = h.mean(dim=2).reshape(B, C * D)
        return self.classifier(feat)

    def _step(self, h, e):
        """한 loop step. recur_mode에 따라 state 갱신."""
        if self.recur_mode == "update":
            f = self.block(h)
            return h + self.inject_norm(f + self.W_inj(e))
        if self.recur_mode == "prenorm":
            f = self.block(self.state_norm(h))
            return h + f + self.W_inj(e)
        # gated: convex combination
        f = self.block(self.state_norm(h))
        cand = f + self.W_inj(e)
        alpha = torch.sigmoid(self.gate_logit)
        return (1.0 - alpha) * h + alpha * cand

    def forward_all_steps(self, x: torch.Tensor, return_norms: bool = False):
        """각 loop step의 분류 logit 리스트 [z_1, ..., z_T] 반환.
        return_norms=True면 (logits, per_step_mean_||h_t||) 도 함께."""
        e, B, C = self._embed(x)
        h = e
        logits = []
        norms = []
        for _ in range(self.loop_T):
            h = self._step(h, e)
            logits.append(self._pool_head(h, B, C))
            if return_norms:
                norms.append(h.norm(dim=-1).mean().item())  # 평균 토큰 ‖h‖
        if return_norms:
            return logits, norms
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_all_steps(x)[-1]

    # train.py의 multi-exit SD 학습 루프(forward_multi_exit 호출)와 인터페이스 통일.
    def forward_multi_exit(self, x: torch.Tensor) -> list:
        return self.forward_all_steps(x)

    @torch.no_grad()
    def forward_dynamic_exit(self, x: torch.Tensor, entropy_thresh: float = 0.3):
        """
        Phase 2 — Dynamic Exit(LoopViT predictive crystallization).
        loop를 돌며 예측 entropy가 임계 미만이면 조기 종료. 추론 전용(분석/효율).
        반환: (logits, used_steps_per_sample)
        """
        e, B, C = self._embed(x)
        h = e
        final = None
        decided = torch.zeros(B, dtype=torch.bool, device=x.device)
        used = torch.full((B,), self.loop_T, dtype=torch.long, device=x.device)
        for t in range(self.loop_T):
            h = self._step(h, e)
            z = self._pool_head(h, B, C)
            if final is None:
                final = z.clone()
            p = torch.softmax(z, dim=-1)
            ent = -(p * torch.log(p + 1e-8)).sum(-1)
            newly = (~decided) & (ent < entropy_thresh)
            final[newly] = z[newly]
            used[newly] = t + 1
            decided = decided | newly
            if decided.all():
                break
        final[~decided] = z[~decided]  # 끝까지 안 정해진 건 마지막 step
        return final, used


class CNNBiLSTM(nn.Module):
    """
    1D-CNN + BiLSTM 시계열 분류 모델.

    Input:  (batch, window, 4)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 12,
        cnn_channels: list[int] = [64, 128],
        cnn_kernel: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # 1D-CNN 블록
        cnn_layers = []
        in_ch = in_channels
        for out_ch in cnn_channels:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, C) → CNN은 (B, C, W) 형식 필요
        x = x.permute(0, 2, 1)          # (B, 4, W)
        x = self.cnn(x)                  # (B, 128, W')
        x = x.permute(0, 2, 1)          # (B, W', 128)
        _, (h, _) = self.lstm(x)         # h: (num_layers*2, B, hidden)
        # 마지막 레이어의 forward/backward hidden 결합
        h = torch.cat([h[-2], h[-1]], dim=1)  # (B, hidden*2)
        return self.classifier(h)


class FFTBranch(nn.Module):
    """
    주파수 도메인 인코더. 입력 (B, W, C)에 대해 채널별 FFT를 구해 인코딩한 뒤
    한 벡터 (B, out_dim)을 반환.

    arch:
      - "cnn"      : FFT magnitude+phase (2C 채널)를 1D CNN으로 인코딩 (기존 기본).
      - "patchtst" : FFT magnitude만 (C 채널) 을 주파수 축을 시퀀스로 보고 PatchTST로
                     인코딩. freq 길이가 매우 짧으므로(W//2+1) overlapping patch 사용.
                     time branch와 동일한 백본을 freq 도메인에 적용한 변형.

    sample_rate_hz: 입력 신호의 샘플링 주파수(Hz). FFT bin → Hz 매핑에 사용.
                    학습/추론에는 영향 없으며 해석·시각화용.
    """

    def __init__(
        self,
        in_channels: int,
        window: int,
        out_dim: int = 128,
        dropout: float = 0.2,
        sample_rate_hz: float = 10.0,
        arch: str = "cnn",
        patchtst_kwargs: dict | None = None,
    ):
        super().__init__()
        self.arch = arch.lower()
        self.out_dim = out_dim
        self.window = window
        self.sample_rate_hz = sample_rate_hz
        self.n_freq = window // 2 + 1

        # FFT bin → Hz 매핑 (buffer로 보관해 state_dict에 같이 저장)
        # rfftfreq(W, d=1/fs) = [0, fs/W, 2fs/W, ..., fs/2]
        freqs_hz = torch.fft.rfftfreq(window, d=1.0 / sample_rate_hz)
        self.register_buffer("freqs_hz", freqs_hz, persistent=False)

        if self.arch == "cnn":
            # mag + phase → 2*C 채널
            self.in_freq_ch = in_channels * 2
            self.conv = nn.Sequential(
                nn.Conv1d(self.in_freq_ch, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(64, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            )
            self.gap = nn.AdaptiveAvgPool1d(1)

        elif self.arch == "patchtst":
            # magnitude only (C 채널), freq 축(n_freq)을 시퀀스로 PatchTST 인코딩.
            kw = dict(patchtst_kwargs or {})
            # freq 시퀀스는 매우 짧으므로 짧은 overlapping patch를 기본값으로.
            kw.setdefault("patch_len", 2)
            kw.setdefault("stride", 1)
            kw.setdefault("d_model", 64)
            kw.setdefault("n_heads", 8)
            kw.setdefault("e_layers", 2)
            kw.setdefault("d_ff", 128)
            kw.setdefault("pool", "mean")
            kw["seq_len"] = self.n_freq
            self.freq_patchtst = PatchTSTClassifier(
                in_channels=in_channels, num_classes=out_dim, **kw
            )
            ln = self.freq_patchtst.classifier[0]  # LayerNorm
            feat_dim = ln.normalized_shape[0]
            # PatchTST feature → out_dim 으로 projection (CNN branch와 동일 차원 유지)
            self.proj = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, out_dim),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"unknown freq arch: {arch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, C)
        # FFT는 마지막 축 기준 → (B, C, W) 형태로 바꾸고 freq dim으로 변환
        xt = x.permute(0, 2, 1)               # (B, C, W)
        Xf = torch.fft.rfft(xt, dim=-1)       # (B, C, W//2+1) complex
        mag = Xf.abs()                        # (B, C, n_freq)

        if self.arch == "cnn":
            phase = torch.angle(Xf)
            feat = torch.cat([mag, phase], dim=1)  # (B, 2C, n_freq)
            feat = self.conv(feat)                 # (B, out_dim, n_freq)
            feat = self.gap(feat).squeeze(-1)      # (B, out_dim)
            return feat

        # patchtst: magnitude (B, C, n_freq) 를 freq-시퀀스로 PatchTST 인코딩
        m = self.freq_patchtst
        feat = m._backbone_features(mag)         # (B, C, d_model, patch_num)
        if m.pool == "flatten":
            feat = feat.flatten(1)               # (B, C*d_model*patch_num)
        else:                                    # mean over patch_num
            feat = feat.mean(dim=-1).flatten(1)  # (B, C*d_model)
        feat = self.proj(feat)                   # (B, out_dim)
        return feat


class HybridTimeFreqClassifier(nn.Module):
    """
    Time branch + Frequency branch를 concat 후 FC로 분류.

    Time branch는 기존 인코더 (cnn_bilstm / resnet1d / patchtst) 중 선택.
    각 인코더의 분류 헤드 직전 representation을 추출해 사용한다.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 12,
        time_arch: str = "resnet1d",
        window: int = 10,
        freq_out_dim: int = 128,
        dropout: float = 0.3,
        sample_rate_hz: float = 10.0,
        # PatchTST 옵션 (time_arch="patchtst"일 때만 사용)
        patchtst_kwargs: dict | None = None,
        # Frequency branch 인코더: "cnn"(기존) | "patchtst"
        freq_arch: str = "cnn",
        freq_patchtst_kwargs: dict | None = None,
    ):
        super().__init__()
        self.time_arch = time_arch.lower()

        # === Time branch === (인코더 + classifier 분리)
        if self.time_arch == "cnn_bilstm":
            # CNNBiLSTM 그대로 만들고, classifier 직전 hidden 크기 추출
            base = CNNBiLSTM(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
            self._time_module = base
            # forward에서 cnn → permute → lstm까지만 통과시킨 뒤 h 결합 사용
            self.time_out_dim = base.lstm.hidden_size * 2

        elif self.time_arch == "resnet1d":
            base = ResNet1D(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
            self._time_module = base
            # ResNet1D 마지막 stage 채널 (256 기본)
            self.time_out_dim = base.classifier.in_features

        elif self.time_arch == "patchtst":
            kw = patchtst_kwargs or {}
            kw.setdefault("seq_len", window)
            base = PatchTSTClassifier(
                in_channels=in_channels, num_classes=num_classes, **kw
            )
            self._time_module = base
            # PatchTSTClassifier의 classifier 첫 LayerNorm입력 차원 사용
            ln = base.classifier[0]  # LayerNorm
            self.time_out_dim = ln.normalized_shape[0]
        else:
            raise ValueError(f"unknown time_arch: {time_arch}")

        # === Frequency branch ===
        self.freq_branch = FFTBranch(
            in_channels=in_channels,
            window=window,
            out_dim=freq_out_dim,
            dropout=dropout,
            sample_rate_hz=sample_rate_hz,
            arch=freq_arch,
            patchtst_kwargs=freq_patchtst_kwargs,
        )

        # === Fusion head ===
        fused_dim = self.time_out_dim + freq_out_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_classes),
        )

    def _time_features(self, x: torch.Tensor) -> torch.Tensor:
        """time encoder의 분류 헤드 직전 feature 반환. (B, time_out_dim)"""
        m = self._time_module
        if self.time_arch == "cnn_bilstm":
            # (B, W, C) → (B, C, W) → CNN → (B, W', 128) → LSTM h
            xt = x.permute(0, 2, 1)
            xt = m.cnn(xt)
            xt = xt.permute(0, 2, 1)
            _, (h, _) = m.lstm(xt)
            h = torch.cat([h[-2], h[-1]], dim=1)  # (B, hidden*2)
            return h

        if self.time_arch == "resnet1d":
            xt = x.permute(0, 2, 1)
            xt = m.stem(xt)
            xt = m.stages(xt)
            xt = m.gap(xt).squeeze(-1)  # (B, C_last)
            return xt

        if self.time_arch == "patchtst":
            xt = x.permute(0, 2, 1)
            feat = m._backbone_features(xt)  # (B, C, d_model, patch_num)
            if m.pool == "flatten":
                feat = feat.flatten(1)
            else:
                feat = feat.mean(dim=-1).flatten(1)
            return feat

        raise RuntimeError(f"unsupported time_arch: {self.time_arch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_feat = self._time_features(x)   # (B, time_out_dim)
        freq_feat = self.freq_branch(x)      # (B, freq_out_dim)
        fused = torch.cat([time_feat, freq_feat], dim=-1)
        return self.classifier(fused)

    def forward_multi_exit(self, x: torch.Tensor) -> list:
        """
        Phase 0(multi-exit self-distillation)용. time branch(PatchTST)의 각 encoder
        layer 출력마다 freq feature와 fusion → step별 logit 리스트 반환.
        리스트 마지막 원소 = 최종 logit(기존 forward와 동일). freq branch는 공유.
        time_arch=patchtst 에서만 의미 있음.
        """
        if self.time_arch != "patchtst":
            return [self.forward(x)]
        m = self._time_module
        xt = x.permute(0, 2, 1)
        layer_feats = m._backbone_features_all_layers(xt)   # list of (B, C, d_model, patch_num)
        freq_feat = self.freq_branch(x)                     # (B, freq_out_dim) — 공유
        logits = []
        for feat in layer_feats:
            tf = m._pool_flatten(feat)                      # (B, time_out_dim)
            fused = torch.cat([tf, freq_feat], dim=-1)
            logits.append(self.classifier(fused))
        return logits


def build_model(num_classes: int = 12, arch: str = "cnn_bilstm", **kwargs) -> nn.Module:
    """
    arch:
      - "cnn_bilstm" : 1D-CNN + BiLSTM (기본)
      - "resnet1d"   : 1D ResNet
      - "patchtst"   : PatchTST 백본 + 분류 헤드 (원본 layers 재사용)
    """
    arch = arch.lower()
    if arch == "resnet1d":
        return ResNet1D(num_classes=num_classes, **kwargs)
    if arch == "cnn_bilstm":
        return CNNBiLSTM(num_classes=num_classes, **kwargs)
    if arch == "patchtst":
        return PatchTSTClassifier(num_classes=num_classes, **kwargs)
    if arch == "cross_patchtst":
        return CrossChannelPatchTSTClassifier(num_classes=num_classes, **kwargs)
    if arch == "looped_patchtst":
        return LoopedPatchTSTClassifier(num_classes=num_classes, **kwargs)
    if arch.startswith("hybrid_"):
        # 예: hybrid_resnet1d, hybrid_cnn_bilstm, hybrid_patchtst
        time_arch = arch[len("hybrid_"):]
        return HybridTimeFreqClassifier(
            num_classes=num_classes, time_arch=time_arch, **kwargs
        )
    raise ValueError(f"unknown arch: {arch}")
