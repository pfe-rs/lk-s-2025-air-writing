from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def compute_ema(values: Iterable[float], alpha: float) -> List[float]:
    ema_values: List[float] = []
    ema: float | None = None
    for value in values:
        ema = value if ema is None else alpha * value + (1 - alpha) * ema
        ema_values.append(ema)
    return ema_values


def plot_loss_acc(
    metrics_csv: Path,
    out_png: Path,
    ema_alpha: float,
) -> None:
    metrics = pd.read_csv(metrics_csv)
    epochs = metrics["epoch"].values
    train_loss = metrics["train_loss"].values
    val_loss = metrics["val_loss"].values
    train_acc_clean = metrics["train_acc_clean"].values
    val_acc = metrics["val_acc"].values
    val_ci_lo = metrics.get("val_ci_lo", pd.Series([np.nan] * len(metrics))).values
    val_ci_hi = metrics.get("val_ci_hi", pd.Series([np.nan] * len(metrics))).values

    val_acc_ema = compute_ema(val_acc, ema_alpha)

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(epochs, train_loss, marker="o", color="red", label="Train loss (clean)")
    ax1.plot(epochs, val_loss, marker="o", color="blue", label="Val loss")
    ax1.set_title("Loss tokom epoha")
    ax1.set_xlabel("Epoha")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(epochs, train_acc_clean, marker="o", color="orange", label="Train acc (clean)")
    ax2.plot(epochs, val_acc, marker="o", color="green", label="Val acc")
    ax2.plot(epochs, val_acc_ema, linestyle="--", color="darkgreen", label=f"Val acc EMA (α={ema_alpha})")

    if not np.isnan(val_ci_lo).all() and not np.isnan(val_ci_hi).all():
        ax2.fill_between(epochs, val_ci_lo, val_ci_hi, color="green", alpha=0.15, label="Val acc 95% CI")

    ax2.set_title("Tačnost tokom epoha")
    ax2.set_xlabel("Epoha")
    ax2.set_ylabel("Tačnost (%)")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_confusion(matrix: np.ndarray, class_names: Sequence[str], out_png: Path) -> None:
    plt.figure(figsize=(8, 7))
    im = plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix (normirano, {len(class_names)} klasa)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.ylabel("Istinita klasa")
    plt.xlabel("Predikcija")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_top_errors(
    images: torch.Tensor,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    out_png: Path,
    max_items: int = 20,
    class_names: Sequence[str] | None = None,
) -> None:
    """Prikazuje najčešće greške (po paru true/pred) i uzima po jednu sliku svakog para."""
    if images.numel() == 0:
        return

    # grupa po (true, pred)
    error_indices = {}
    counts = {}
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            continue
        key = (int(t), int(p))
        counts[key] = counts.get(key, 0) + 1
        if key not in error_indices:
            error_indices[key] = idx

    if not counts:
        return

    sorted_keys = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)[:max_items]
    selected_indices = [error_indices[k] for k in sorted_keys]

    selected_images = images[selected_indices]
    selected_true = [y_true[i] for i in selected_indices]
    selected_pred = [y_pred[i] for i in selected_indices]

    cols = 5
    rows = max(1, int(np.ceil(len(selected_images) / cols)))
    plt.figure(figsize=(3 * cols, 3 * rows))
    for idx, (img, t, p) in enumerate(zip(selected_images, selected_true, selected_pred)):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img.squeeze(0), cmap="gray")
        true_label = class_names[t] if class_names else str(t)
        pred_label = class_names[p] if class_names else str(p)
        ax.set_title(f"T: {true_label}\nP: {pred_label}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
