"""
시계열 신호용 online augmentation.

기본 4종 (TimeSeriesNoise):
  - Gaussian: ADC/회로 thermal noise
  - amplitude jitter: 배터리 잔량/온도 변동
  - DC bias: 센서 calibration drift
  - spike: 모터 PWM, EMI 등 impulse noise

강화 종합 패키지 (RealEnvNoise):
  위 4종 + drift / pink noise / hum / mixup / masking / 경계 침범 bias
"""

import numpy as np
import torch


class TimeSeriesNoise:
    """학습 시점에 적용되는 노이즈 변환. eval/test에서는 비활성화."""

    def __init__(
        self,
        gaussian_std: float = 0.05,      # Gaussian 표준편차 (정규화 후 값 단위)
        amp_jitter: float = 0.05,        # 진폭 스케일 ±비율 (0.05 = ±5%)
        bias_std: float = 0.05,          # DC offset 표준편차
        spike_prob: float = 0.01,        # 각 sample에 spike 발생 확률
        spike_std: float = 0.3,          # spike 크기 표준편차
        p_apply: float = 1.0,            # 윈도우별 노이즈 적용 확률
        enabled: bool = True,
    ):
        self.gaussian_std = gaussian_std
        self.amp_jitter = amp_jitter
        self.bias_std = bias_std
        self.spike_prob = spike_prob
        self.spike_std = spike_std
        self.p_apply = p_apply
        self.enabled = enabled

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (W, C) — 한 윈도우. dtype float32, 정규화된 값.
        반환: 같은 shape의 노이즈가 더해진 tensor.
        """
        if not self.enabled or np.random.random() > self.p_apply:
            return x

        W, C = x.shape
        # 모든 노이즈는 같은 dtype/device 유지
        out = x.clone()

        # 1) 진폭 jitter — 채널별 동일 스케일 (전체 신호가 함께 커짐/작아짐)
        if self.amp_jitter > 0:
            scale = 1.0 + (np.random.random() * 2 - 1) * self.amp_jitter
            out = out * scale

        # 2) DC bias — 채널별 독립 오프셋
        if self.bias_std > 0:
            bias = torch.randn(C, dtype=out.dtype) * self.bias_std
            out = out + bias.unsqueeze(0)  # (1, C) 브로드캐스트

        # 3) Gaussian — 모든 sample 독립
        if self.gaussian_std > 0:
            out = out + torch.randn_like(out) * self.gaussian_std

        # 4) Spike — drop-in impulse, 위치는 랜덤
        if self.spike_prob > 0 and self.spike_std > 0:
            mask = (torch.rand_like(out) < self.spike_prob).float()
            spikes = torch.randn_like(out) * self.spike_std * mask
            out = out + spikes

        return out

    def __repr__(self):
        return (
            f"TimeSeriesNoise(g={self.gaussian_std}, "
            f"amp=±{self.amp_jitter}, bias={self.bias_std}, "
            f"spike={self.spike_prob}@{self.spike_std}, enabled={self.enabled})"
        )


def _pink_noise_1d(n: int, dtype=torch.float32) -> torch.Tensor:
    """길이 n의 1/f (pink) 노이즈 한 줄. unit variance로 정규화."""
    n_freqs = n // 2 + 1
    freqs = np.arange(n_freqs).astype(np.float32)
    freqs[0] = 1.0  # DC 발산 방지
    spectrum = 1.0 / np.sqrt(freqs)
    phases = np.random.uniform(0, 2 * np.pi, n_freqs)
    complex_spec = spectrum * np.exp(1j * phases)
    sig = np.fft.irfft(complex_spec, n=n).astype(np.float32)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    return torch.from_numpy(sig).to(dtype)


class RealEnvNoise:
    """
    실 환경 노이즈 종합 패키지. 기본 4종 + 다음을 모두 포함:
      - drift: 윈도우 길이 안에서 천천히 변하는 저주파 offset (sin)
      - pink: 1/f 스펙트럼을 가진 thermal/flicker noise
      - hum: 60Hz 또는 사용자 지정 주파수의 sin 간섭
      - boundary bias: 클래스 경계(저/고부하)를 침범할 만큼 큰 ±offset
      - masking: 일정 비율의 시간 구간을 0으로 덮음 (센서 dropout 모사)
      - mixup: 다른 윈도우와 lambda 비율로 섞음. Dataset이 mixup 페어를 따로 제공해야 함.
              (간단화를 위해 같은 배치 내가 아니라 같은 윈도우 인덱스의 noise만 적용)

    Mixup은 외부에서 페어 윈도우를 넘겨주는 방식이 정석이지만, 여기선 호출부 변경
    부담을 줄이기 위해 __call__이 단일 윈도우만 받도록 두고 mixup은 비활성화.
    필요 시 train.py 쪽에서 batch-level mixup을 추가하는 게 깔끔함.
    """

    def __init__(
        self,
        # 기본 4종
        gaussian_std: float = 0.1,
        amp_jitter: float = 0.10,
        bias_std: float = 0.05,
        spike_prob: float = 0.02,
        spike_std: float = 0.5,
        # 강화 항목
        drift_amp: float = 0.5,            # 정규화 단위 (z-score 0.5 ≈ 클래스 경계 부근 흔듦)
        drift_periods: tuple = (0.5, 2.0), # 윈도우 길이 대비 주기 (0.5=2배 길이, 2=절반)
        pink_std: float = 0.1,
        hum_amp: float = 0.05,
        hum_freq_hz: float = 60.0,
        sample_rate_hz: float = 10.0,      # 데이터 샘플링 (0.1초 간격 = 10Hz)
        boundary_bias_prob: float = 0.3,   # 큰 bias 적용 확률
        boundary_bias_max: float = 0.7,    # 정규화 단위 최대 ±
        mask_prob: float = 0.1,            # 마스킹 발생 확률
        mask_ratio: float = 0.2,           # 마스킹될 길이 비율
        p_apply: float = 1.0,
        enabled: bool = True,
    ):
        # 기본
        self.gaussian_std = gaussian_std
        self.amp_jitter = amp_jitter
        self.bias_std = bias_std
        self.spike_prob = spike_prob
        self.spike_std = spike_std
        # 강화
        self.drift_amp = drift_amp
        self.drift_periods = drift_periods
        self.pink_std = pink_std
        self.hum_amp = hum_amp
        self.hum_freq_hz = hum_freq_hz
        self.sample_rate_hz = sample_rate_hz
        self.boundary_bias_prob = boundary_bias_prob
        self.boundary_bias_max = boundary_bias_max
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        self.p_apply = p_apply
        self.enabled = enabled

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or np.random.random() > self.p_apply:
            return x

        W, C = x.shape
        out = x.clone()

        # 1) 진폭 jitter
        if self.amp_jitter > 0:
            scale = 1.0 + (np.random.random() * 2 - 1) * self.amp_jitter
            out = out * scale

        # 2) DC bias (작은)
        if self.bias_std > 0:
            bias = torch.randn(C, dtype=out.dtype) * self.bias_std
            out = out + bias.unsqueeze(0)

        # 2b) 경계 침범 bias (큰) — 일정 확률로 적용
        if self.boundary_bias_prob > 0 and np.random.random() < self.boundary_bias_prob:
            big_bias = (np.random.random() * 2 - 1) * self.boundary_bias_max
            out = out + big_bias

        # 3) Drift — 채널별 sin 파형. 주기는 윈도우 길이 대비 랜덤
        if self.drift_amp > 0:
            t = torch.arange(W, dtype=out.dtype) / W
            for c in range(C):
                period_ratio = np.random.uniform(*self.drift_periods)
                phase = np.random.uniform(0, 2 * np.pi)
                drift = self.drift_amp * np.random.uniform(-1, 1) * \
                    torch.sin(2 * np.pi / period_ratio * t + phase)
                out[:, c] = out[:, c] + drift

        # 4) Pink noise (channel-independent)
        if self.pink_std > 0:
            for c in range(C):
                pn = _pink_noise_1d(W, dtype=out.dtype) * self.pink_std
                out[:, c] = out[:, c] + pn

        # 5) Hum — 모든 채널에 같은 60Hz (혹은 다른 freq) sin
        # Nyquist 한계: 10Hz 샘플링에서는 60Hz를 직접 못 잡으니 aliasing 주파수 사용
        if self.hum_amp > 0:
            t_sec = torch.arange(W, dtype=out.dtype) / self.sample_rate_hz
            hum = self.hum_amp * torch.sin(2 * np.pi * self.hum_freq_hz * t_sec)
            out = out + hum.unsqueeze(1)

        # 6) Gaussian (white)
        if self.gaussian_std > 0:
            out = out + torch.randn_like(out) * self.gaussian_std

        # 7) Spike
        if self.spike_prob > 0 and self.spike_std > 0:
            mask = (torch.rand_like(out) < self.spike_prob).float()
            spikes = torch.randn_like(out) * self.spike_std * mask
            out = out + spikes

        # 8) Masking — 연속 구간 0
        if self.mask_prob > 0 and np.random.random() < self.mask_prob:
            mask_len = int(W * self.mask_ratio)
            if mask_len > 0:
                start = np.random.randint(0, W - mask_len + 1)
                out[start:start + mask_len, :] = 0.0

        return out

    def __repr__(self):
        return (
            f"RealEnvNoise(g={self.gaussian_std}, amp=±{self.amp_jitter}, "
            f"bias={self.bias_std}, spike={self.spike_prob}@{self.spike_std}, "
            f"drift={self.drift_amp}, pink={self.pink_std}, hum={self.hum_amp}@{self.hum_freq_hz}Hz, "
            f"bbias={self.boundary_bias_prob}@±{self.boundary_bias_max}, "
            f"mask={self.mask_prob}@{self.mask_ratio}, enabled={self.enabled})"
        )


class DomainRandomNoise:
    """
    실측 데이터 없이 sim-to-real gap을 줄이기 위한 domain randomization.

    핵심 — 매 윈도우마다 노이즈 환경 자체가 랜덤하게 바뀐다. 일관된 노이즈가 아니라
    각 윈도우가 서로 다른 측정 조건에서 측정된 것처럼 보임.

    5가지 구조:
      1. AR(1) 자기상관 노이즈 (시간상 부드러운 노이즈)
      2. 채널간 상관 노이즈 (한 노이즈가 모든 채널에 함께 들어감 + 채널별 독립 노이즈)
      3. 윈도우별 SNR 랜덤화 (매 호출마다 노이즈 강도 자체가 ×0.5~×3 변동)
      4. 랜덤 LP smoothing (신호 모양 자체를 약하게 흔듦)
      5. 간헐적 saturation (한 윈도우 전체를 max로 clip)
    """

    def __init__(
        self,
        base_noise_std: float = 0.1,         # 기본 노이즈 σ (정규화 단위)
        snr_range: tuple = (0.5, 3.0),       # 윈도우별 노이즈 강도 배율 범위
        ar1_phi_range: tuple = (0.5, 0.95),  # AR(1) 자기상관 계수
        cross_channel_ratio: float = 0.5,    # 공통 노이즈 : 채널별 독립 노이즈 비율
        smooth_prob: float = 0.5,            # LP smoothing 적용 확률
        smooth_kernel_range: tuple = (3, 5), # smoothing 커널 크기 범위 (홀수)
        sat_prob: float = 0.02,              # saturation 발생 확률
        sat_clip: float = 1.5,               # clip 범위 (±, 정규화 단위)
        bias_drift_std: float = 0.3,         # 윈도우별 큰 drift (느린 변동)
        quantize_prob: float = 0.3,          # quantization 적용 확률
        quantize_bits_range: tuple = (8, 12),# quantization bit 깊이 범위
        p_apply: float = 1.0,
        enabled: bool = True,
    ):
        self.base_noise_std = base_noise_std
        self.snr_range = snr_range
        self.ar1_phi_range = ar1_phi_range
        self.cross_channel_ratio = cross_channel_ratio
        self.smooth_prob = smooth_prob
        self.smooth_kernel_range = smooth_kernel_range
        self.sat_prob = sat_prob
        self.sat_clip = sat_clip
        self.bias_drift_std = bias_drift_std
        self.quantize_prob = quantize_prob
        self.quantize_bits_range = quantize_bits_range
        self.p_apply = p_apply
        self.enabled = enabled

    def _ar1_noise(self, W: int, phi: float, sigma: float, dtype) -> torch.Tensor:
        """AR(1) 노이즈 한 줄. x_t = phi * x_{t-1} + eps_t."""
        eps = torch.randn(W, dtype=dtype) * sigma * np.sqrt(1 - phi * phi)
        x = torch.zeros(W, dtype=dtype)
        x[0] = eps[0]
        for t in range(1, W):
            x[t] = phi * x[t - 1] + eps[t]
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or np.random.random() > self.p_apply:
            return x

        W, C = x.shape
        out = x.clone()

        # === 1. 신호 자체 약한 smoothing (측정 시 cable LP 효과) ===
        if np.random.random() < self.smooth_prob:
            k_min, k_max = self.smooth_kernel_range
            k = np.random.randint(k_min, k_max + 1) // 2 * 2 + 1  # 홀수
            kernel = torch.ones(C, 1, k, dtype=out.dtype) / k
            xt = out.transpose(0, 1).unsqueeze(0)  # (1, C, W)
            xt = torch.nn.functional.conv1d(xt, kernel, padding=k // 2, groups=C)
            out = xt.squeeze(0).transpose(0, 1)

        # === 2. 윈도우별 SNR 랜덤화 ===
        snr_scale = np.random.uniform(*self.snr_range)
        sigma = self.base_noise_std * snr_scale

        # === 3. AR(1) 노이즈 - 자기상관 ===
        phi = np.random.uniform(*self.ar1_phi_range)

        # 공통 노이즈 (모든 채널에 같은 패턴이 들어감 - 케이블/EMI 전달성)
        shared = self._ar1_noise(W, phi, sigma * self.cross_channel_ratio, out.dtype)

        # 채널별 독립 AR(1) 노이즈
        for c in range(C):
            indep_phi = np.random.uniform(*self.ar1_phi_range)
            indep = self._ar1_noise(W, indep_phi, sigma * (1 - self.cross_channel_ratio), out.dtype)
            out[:, c] = out[:, c] + shared + indep

        # === 4. 윈도우 단위 큰 bias drift (매 측정 환경이 다름) ===
        if self.bias_drift_std > 0:
            bias = torch.randn(C, dtype=out.dtype) * self.bias_drift_std
            out = out + bias.unsqueeze(0)

        # === 5. Random quantization (ADC 다양성) ===
        if np.random.random() < self.quantize_prob:
            b_min, b_max = self.quantize_bits_range
            bits = np.random.randint(b_min, b_max + 1)
            levels = 2 ** bits
            v_min, v_max = out.min().item(), out.max().item()
            v_range = max(v_max - v_min, 1e-6)
            step = v_range / levels
            out = torch.round((out - v_min) / step) * step + v_min

        # === 6. 간헐적 saturation ===
        if np.random.random() < self.sat_prob:
            # 전체 윈도우의 일부 채널을 clip
            ch = np.random.randint(0, C)
            out[:, ch] = torch.clamp(out[:, ch], -self.sat_clip, self.sat_clip)

        return out

    def __repr__(self):
        return (
            f"DomainRandomNoise(base_σ={self.base_noise_std}, "
            f"snr×{self.snr_range}, ar1_phi={self.ar1_phi_range}, "
            f"cross={self.cross_channel_ratio}, "
            f"smooth_p={self.smooth_prob}, sat_p={self.sat_prob}, "
            f"bias_drift={self.bias_drift_std}, "
            f"quant_p={self.quantize_prob}@bits{self.quantize_bits_range}, "
            f"enabled={self.enabled})"
        )
