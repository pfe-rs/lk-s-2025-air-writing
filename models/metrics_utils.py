import math
from typing import Iterable, Tuple

import numpy as np
import torch


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval (u procentima)."""
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denominator = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    return lower * 100.0, upper * 100.0


def compute_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    num_classes: int,
    normalize: bool = True,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        matrix[int(t), int(p)] += 1
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums
    return matrix


def unnormalize_images(tensors: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Vrati slike nazad u [0,1] opseg (pretpostavlja mean/std skalare)."""
    return tensors * std + mean
