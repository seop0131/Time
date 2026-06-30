import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def get_class_names(num_classes: int) -> list[str]:
    """이진(저/고부하) / 11-class (merge11) / 12-class 라벨 이름 반환."""
    if num_classes == 2:
        return ["low_load", "high_load"]
    if num_classes == 11:
        # merge11: label 11 → 0으로 합쳐 0이 휴식, 1~10 그대로
        return ["rest"] + [f"label_{i}" for i in range(1, 11)]
    return [f"label_{i}" for i in range(num_classes)]


def compute_metrics(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> dict:
    """예측 결과에서 분류 지표 일괄 계산."""
    class_ids = list(range(num_classes))
    class_names = get_class_names(num_classes)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, labels=class_ids, average="weighted", zero_division=0)

    # per-class precision / recall / f1 / support
    p, r, f1_per, support = precision_recall_fscore_support(
        labels, preds, labels=class_ids, zero_division=0
    )

    per_class = {}
    for i in class_ids:
        per_class[class_names[i]] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1_per[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "per_class": per_class,
    }


def print_metrics(m: dict, title: str = "Metrics") -> None:
    """compute_metrics 결과를 표 형태로 출력."""
    print(f"\n=== {title} ===")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  F1 score : {m['f1']:.4f}")

    print(f"\n  {'class':<12}{'precision':>11}{'recall':>10}{'f1':>10}{'support':>10}")
    for name, c in m["per_class"].items():
        print(
            f"  {name:<12}{c['precision']:>11.4f}{c['recall']:>10.4f}"
            f"{c['f1']:>10.4f}{c['support']:>10d}"
        )


def save_confusion_matrix(
    labels: np.ndarray, preds: np.ndarray, num_classes: int, path: str
) -> None:
    """confusion matrix를 정규화 히트맵 png로 저장."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib 없음 - confusion matrix png 생략)")
        return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    class_ids = list(range(num_classes))
    class_names = get_class_names(num_classes)
    cm = confusion_matrix(labels, preds, labels=class_ids)
    # row 정규화 (각 실제 클래스 기준 비율)
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(class_ids)
    ax.set_yticks(class_ids)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")

    for i in class_ids:
        for j in class_ids:
            val = cm_norm[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="white" if val > 0.5 else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  confusion matrix 저장: {path}")


def save_metrics_json(m: dict, path: str) -> None:
    """지표 dict를 json으로 저장."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"  metrics json 저장: {path}")
