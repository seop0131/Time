import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_COLS = ["VBAT", "IBAT", "ICHG", "ILOAD"]
LABEL_COL = "motion_label"
SAMPLE_RATE_HZ = 10.0      # 0.1s 간격 샘플링 = 10Hz
SAMPLE_PERIOD_S = 1.0 / SAMPLE_RATE_HZ
STEPS_PER_HOUR = 36_000   # 0.1s 간격, 1시간 = 36000 스텝

# 이진분류용 매핑: 저부하=0, 고부하=1
HIGH_LOAD_LABELS = {2, 3, 4, 5, 6}
LOW_LOAD_LABELS = {0, 1, 7, 8, 9, 10, 11}


def to_binary_labels(labels: np.ndarray) -> np.ndarray:
    """12-class motion_label을 이진(0=저부하, 1=고부하)으로 변환."""
    binary = np.zeros_like(labels, dtype=np.int64)
    for hl in HIGH_LOAD_LABELS:
        binary[labels == hl] = 1
    return binary


def to_merged11_labels(labels: np.ndarray) -> np.ndarray:
    """
    12-class → 11-class: label 11(동작 무부하)을 label 0(휴식)으로 통합.
    신호상 거의 동일한 두 휴식 상태를 하나의 클래스로 본다.
    결과 라벨 공간: {0(휴식), 1~10(동작)}.
    """
    merged = labels.copy()
    merged[merged == 11] = 0
    return merged


class SlidingWindowDataset(Dataset):
    transform = None  # 클래스 기본값. __new__로 만든 subset에서도 None으로 안전

    def __init__(self, data: np.ndarray, labels: np.ndarray, window: int, stride: int):
        """
        data:   (N, 4) float32
        labels: (N,)   int64
        """
        self.window = window
        self.samples = []
        self.targets = []

        for start in range(0, len(data) - window + 1, stride):
            end = start + window
            window_labels = labels[start:end]

            # 경계 윈도우 제거: 윈도우 안에 2개 이상 label이 섞이면 스킵
            if len(np.unique(window_labels)) > 1:
                continue

            self.samples.append(data[start:end])
            self.targets.append(window_labels[-1])

        self.samples = np.stack(self.samples).astype(np.float32)  # (M, W, 4)
        self.targets = np.array(self.targets, dtype=np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx])
        if self.transform is not None:
            x = self.transform(x)
        return x, torch.tensor(self.targets[idx])


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    data = df[FEATURE_COLS].values.astype(np.float32)
    labels = df[LABEL_COL].values.astype(np.int64)
    return data, labels


def normalize(train_data: np.ndarray, *others: np.ndarray):
    """train 통계로 z-score 정규화. train + 나머지 배열 반환."""
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0) + 1e-8
    normalized = [(d - mean) / std for d in (train_data, *others)]
    return normalized, mean, std


def make_splits(
    csv_path: str,
    window: int = 100,
    stride: int = 50,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    binary: bool = False,
) -> tuple[SlidingWindowDataset, SlidingWindowDataset, SlidingWindowDataset]:
    """
    1시간 블록(36000 스텝) 단위로 train/val/test split.
    시간 순서를 유지해 data leakage 방지.
    binary=True면 12-class label을 저/고부하 이진으로 매핑.
    """
    data, labels = load_csv(csv_path)
    if binary:
        labels = to_binary_labels(labels)
    n_hours = len(data) // STEPS_PER_HOUR

    n_train = int(n_hours * train_ratio)
    n_val = int(n_hours * val_ratio)

    train_end = n_train * STEPS_PER_HOUR
    val_end = (n_train + n_val) * STEPS_PER_HOUR

    train_data, train_labels = data[:train_end], labels[:train_end]
    val_data, val_labels = data[train_end:val_end], labels[train_end:val_end]
    test_data, test_labels = data[val_end:], labels[val_end:]

    (train_data, val_data, test_data), mean, std = normalize(train_data, val_data, test_data)

    train_ds = SlidingWindowDataset(train_data, train_labels, window, stride)
    val_ds = SlidingWindowDataset(val_data, val_labels, window, stride)
    test_ds = SlidingWindowDataset(test_data, test_labels, window, stride)

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    return train_ds, val_ds, test_ds


def extract_first_per_label(
    data: np.ndarray, labels: np.ndarray, samples_per_label: int = 600
) -> tuple[np.ndarray, np.ndarray]:
    """
    각 label이 처음 등장하는 지점부터 연속된 samples_per_label 행을 추출해 이어붙임.
    반환되는 시퀀스는 label 순서로 정렬된 (samples_per_label × num_labels, ...) 형태.
    """
    unique_labels = sorted(np.unique(labels).tolist())
    chunks_data, chunks_labels = [], []
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        if len(idx) < samples_per_label:
            raise ValueError(f"label {lab}: 데이터 {len(idx)}행으로 {samples_per_label}행에 부족")
        start = int(idx[0])
        end = start + samples_per_label
        chunks_data.append(data[start:end])
        chunks_labels.append(labels[start:end])
    return np.concatenate(chunks_data, axis=0), np.concatenate(chunks_labels, axis=0)


def make_splits_first_per_label(
    csv_path: str,
    samples_per_label: int = 600,
    window: int = 10,
    stride: int = 10,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    binary: bool = False,
    seed: int = 42,
    split_mode: str = "random",
    trim_head: int = 0,
    trim_tail: int = 0,
    merge11: bool = False,
) -> tuple[SlidingWindowDataset, SlidingWindowDataset, SlidingWindowDataset]:
    """
    각 label의 첫 등장 구간 samples_per_label 행만 추출 후 split.

    split_mode:
      - "random"   : 윈도우 생성 후 윈도우 인덱스를 셔플해 80/10/10. 인접 윈도우가 split에 흩어져 누수 가능.
      - "temporal" : 클래스별 행을 시간순 80/10/10으로 나눈 뒤 각 split에서 윈도우 생성. 인접 윈도우가 한 split에만 모임.
    """
    data, labels = load_csv(csv_path)
    # binary 매핑은 split 이후에 적용 (원본 12-class 기준으로 클래스별 600행을 잘라야 정확함)
    data, labels = extract_first_per_label(data, labels, samples_per_label)

    # 각 클래스 구간의 첫 trim_head 행과 마지막 trim_tail 행 제거 (transient 구간 제외)
    if trim_head > 0 or trim_tail > 0:
        if trim_head + trim_tail >= samples_per_label:
            raise ValueError(
                f"trim_head({trim_head}) + trim_tail({trim_tail}) >= samples_per_label({samples_per_label})"
            )
        effective = samples_per_label - trim_head - trim_tail
        unique_orig = sorted(np.unique(labels).tolist())
        keep_d, keep_l = [], []
        for i, _ in enumerate(unique_orig):
            base = i * samples_per_label
            start = base + trim_head
            end = base + samples_per_label - trim_tail
            keep_d.append(data[start:end])
            keep_l.append(labels[start:end])
        data = np.concatenate(keep_d, axis=0)
        labels = np.concatenate(keep_l, axis=0)
        # 이후 코드가 samples_per_label을 클래스 구간 길이로 사용하므로 갱신
        samples_per_label = effective

    orig_labels = labels  # 원본 12-class 보관 (temporal split에서 사용)
    if binary:
        labels = to_binary_labels(labels)
    elif merge11:
        # 11(동작 무부하) → 0(휴식). 라벨 공간 {0, 1~10}
        labels = to_merged11_labels(labels)

    # 추출된 데이터 통계로 정규화 (전체 = train+val+test이므로 동일)
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8
    data = (data - mean) / std

    if split_mode == "random":
        full_ds = SlidingWindowDataset(data, labels, window, stride)
        n = len(full_ds)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]

        def subset(idx):
            sub = SlidingWindowDataset.__new__(SlidingWindowDataset)
            sub.window = window
            sub.samples = full_ds.samples[idx]
            sub.targets = full_ds.targets[idx]
            return sub

        train_ds, val_ds, test_ds = subset(train_idx), subset(val_idx), subset(test_idx)
        mode_desc = "random window split"

    elif split_mode == "temporal":
        # 각 클래스 samples_per_label 안에서 시간순 80/10/10
        # extract_first_per_label이 원본 12-class 순서로 정렬해 이어붙였으므로,
        # 그 순서(orig_labels의 unique 순서)를 기준으로 클래스별 블록을 잘라야 함
        n_train_rows = int(samples_per_label * train_ratio)
        n_val_rows = int(samples_per_label * val_ratio)

        unique_orig = sorted(np.unique(orig_labels).tolist())
        train_chunks_d, train_chunks_l = [], []
        val_chunks_d, val_chunks_l = [], []
        test_chunks_d, test_chunks_l = [], []
        for i, _ in enumerate(unique_orig):
            base = i * samples_per_label
            d_lab = data[base : base + samples_per_label]
            l_lab = labels[base : base + samples_per_label]
            train_chunks_d.append(d_lab[:n_train_rows])
            train_chunks_l.append(l_lab[:n_train_rows])
            val_chunks_d.append(d_lab[n_train_rows : n_train_rows + n_val_rows])
            val_chunks_l.append(l_lab[n_train_rows : n_train_rows + n_val_rows])
            test_chunks_d.append(d_lab[n_train_rows + n_val_rows :])
            test_chunks_l.append(l_lab[n_train_rows + n_val_rows :])

        train_data = np.concatenate(train_chunks_d, axis=0)
        train_labels = np.concatenate(train_chunks_l, axis=0)
        val_data = np.concatenate(val_chunks_d, axis=0)
        val_labels = np.concatenate(val_chunks_l, axis=0)
        test_data = np.concatenate(test_chunks_d, axis=0)
        test_labels = np.concatenate(test_chunks_l, axis=0)

        train_ds = SlidingWindowDataset(train_data, train_labels, window, stride)
        val_ds = SlidingWindowDataset(val_data, val_labels, window, stride)
        test_ds = SlidingWindowDataset(test_data, test_labels, window, stride)
        mode_desc = "temporal split (per-label, time-ordered)"

    else:
        raise ValueError(f"unknown split_mode: {split_mode}")

    print(
        f"[first-per-label / {mode_desc}] {csv_path}\n"
        f"  per label = {samples_per_label} rows, total = {len(data):,} rows\n"
        f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}"
    )
    return train_ds, val_ds, test_ds


def make_splits_random_two_csv(
    train_csv: str,
    test_csv: str,
    window: int = 100,
    stride: int = 50,
    val_ratio: float = 0.1,
    binary: bool = False,
    seed: int = 42,
) -> tuple[SlidingWindowDataset, SlidingWindowDataset, SlidingWindowDataset]:
    """
    train_csv를 윈도우 단위 random으로 train/val로 나누고, test_csv 전체를 test로 사용.
    - 정규화 통계는 train_csv 데이터로 추정해 test_csv에도 동일 적용
    - train/val은 같은 csv 내에서 셔플되므로 같은 1분 구간 윈도우가 섞일 수 있음(의도된 진단)
    - test는 별도 파일이므로 train과 윈도우 단위 누수는 없음
    """
    train_full_data, train_full_labels = load_csv(train_csv)
    test_data, test_labels = load_csv(test_csv)
    if binary:
        train_full_labels = to_binary_labels(train_full_labels)
        test_labels = to_binary_labels(test_labels)

    # train_csv 통계로 정규화
    mean = train_full_data.mean(axis=0)
    std = train_full_data.std(axis=0) + 1e-8
    train_full_data = (train_full_data - mean) / std
    test_data = (test_data - mean) / std

    # 슬라이딩 윈도우 생성 후 셔플
    full_ds = SlidingWindowDataset(train_full_data, train_full_labels, window, stride)
    n = len(full_ds)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_val = int(n * val_ratio)
    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    def subset(idx):
        sub = SlidingWindowDataset.__new__(SlidingWindowDataset)
        sub.window = window
        sub.samples = full_ds.samples[idx]
        sub.targets = full_ds.targets[idx]
        return sub

    train_ds, val_ds = subset(train_idx), subset(val_idx)
    test_ds = SlidingWindowDataset(test_data, test_labels, window, stride)

    print(
        f"[random-split + two-csv] train={train_csv}  test={test_csv}\n"
        f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}"
    )
    return train_ds, val_ds, test_ds


def make_splits_random(
    csv_path: str,
    window: int = 100,
    stride: int = 50,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    binary: bool = False,
    seed: int = 42,
) -> tuple[SlidingWindowDataset, SlidingWindowDataset, SlidingWindowDataset]:
    """
    [진단용] 윈도우 단위 완전 random split.
    같은 1분 증강 구간의 윈도우들이 train/val/test에 흩어질 수 있어 누수 발생.
    정규화는 전체 데이터 통계로 적용(시간 분할 의미가 없으므로).
    """
    data, labels = load_csv(csv_path)
    if binary:
        labels = to_binary_labels(labels)

    # 전체 통계로 정규화
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8
    data = (data - mean) / std

    # 한 번에 모든 윈도우를 만들고 인덱스로 shuffle
    full_ds = SlidingWindowDataset(data, labels, window, stride)
    n = len(full_ds)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    def subset(idx):
        sub = SlidingWindowDataset.__new__(SlidingWindowDataset)
        sub.window = window
        sub.samples = full_ds.samples[idx]
        sub.targets = full_ds.targets[idx]
        return sub

    train_ds, val_ds, test_ds = subset(train_idx), subset(val_idx), subset(test_idx)
    print(
        f"[random-split] WARNING: 같은 1분 구간의 윈도우가 split에 흩어집니다 (누수 가능)\n"
        f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}"
    )
    return train_ds, val_ds, test_ds


def make_splits_two_csv(
    train_csv: str,
    test_csv: str,
    window: int = 100,
    stride: int = 50,
    val_ratio: float = 0.1,
    binary: bool = False,
) -> tuple[SlidingWindowDataset, SlidingWindowDataset, SlidingWindowDataset]:
    """
    train_csv를 train+val로, test_csv 전체를 test로 사용.
    - train_csv의 뒤쪽 val_ratio 만큼을 val로 분리 (1시간 블록 단위, 시간 순서 유지)
    - 정규화 통계는 train 데이터에서만 추정하고 val/test에도 동일 적용
    - binary=True면 12-class label을 저/고부하 이진으로 매핑
    """
    train_full_data, train_full_labels = load_csv(train_csv)
    test_data, test_labels = load_csv(test_csv)
    if binary:
        train_full_labels = to_binary_labels(train_full_labels)
        test_labels = to_binary_labels(test_labels)

    n_hours = len(train_full_data) // STEPS_PER_HOUR
    n_val = int(n_hours * val_ratio)
    n_train = n_hours - n_val

    train_end = n_train * STEPS_PER_HOUR
    val_end = (n_train + n_val) * STEPS_PER_HOUR

    train_data = train_full_data[:train_end]
    train_labels = train_full_labels[:train_end]
    val_data = train_full_data[train_end:val_end]
    val_labels = train_full_labels[train_end:val_end]

    (train_data, val_data, test_data), mean, std = normalize(train_data, val_data, test_data)

    train_ds = SlidingWindowDataset(train_data, train_labels, window, stride)
    val_ds = SlidingWindowDataset(val_data, val_labels, window, stride)
    test_ds = SlidingWindowDataset(test_data, test_labels, window, stride)

    print(
        f"[two-csv] train={train_csv}  test={test_csv}\n"
        f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}"
    )
    return train_ds, val_ds, test_ds
