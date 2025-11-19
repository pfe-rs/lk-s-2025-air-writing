from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DEFAULT_NORMALIZE_MEAN = 0.1736
DEFAULT_NORMALIZE_STD = 0.3249


class EMNISTDataset(Dataset):
    """Dataset koji učitava EMNIST CSV (Letters varijanta) i vraća u tensor obliku."""

    def __init__(
        self,
        csv_path: Path,
        augment: bool,
        mean: float = DEFAULT_NORMALIZE_MEAN,
        std: float = DEFAULT_NORMALIZE_STD,
        random_erasing_p: float = 0.0,
    ):
        if not csv_path.exists():
            raise FileNotFoundError(f"EMNIST CSV nije pronađen: {csv_path}")

        df = pd.read_csv(csv_path)
        self.images = df.iloc[:, 1:].values.astype(np.uint8)
        # EMNIST Letters labele su u rasponu [1,26] -> pomeramo na [0,25]
        self.labels = df.iloc[:, 0].values.astype(np.int64) - 1
        self.transform = self._build_transform(augment, mean, std, random_erasing_p)

    @staticmethod
    def _build_transform(augment: bool, mean: float, std: float, random_erasing_p: float):
        ops: List[transforms.Compose] = [transforms.ToPILImage()]
        if augment:
            ops.append(
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.05, 0.05),
                    shear=5,
                )
            )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[mean], std=[std]),
            ]
        )
        if augment and random_erasing_p > 0.0:
            ops.append(
                transforms.RandomErasing(
                    p=random_erasing_p,
                    scale=(0.02, 0.15),
                    ratio=(0.3, 3.3),
                )
            )
        return transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.labels)

    @staticmethod
    def _reshape(image_row: np.ndarray) -> np.ndarray:
        img = image_row.reshape(28, 28)
        # EMNIST je “naopako”: rotacija + flip da bi slova imala isti orijentir kao ručno segmentirane slike
        img = np.rot90(img, k=1)
        img = np.flipud(img)
        return img

    def __getitem__(self, index: int):
        img = self._reshape(self.images[index])
        tensor_img = self.transform(img)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return tensor_img, label


def create_dataloaders(
    train_csv: Path,
    val_csv: Path,
    batch_size: int,
    num_workers: int,
    mean: float = DEFAULT_NORMALIZE_MEAN,
    std: float = DEFAULT_NORMALIZE_STD,
    random_erasing_p: float = 0.1,
    disable_augmentations: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Vraća (train_aug_loader, train_clean_loader, val_loader)."""
    train_aug_dataset = EMNISTDataset(
        train_csv,
        augment=not disable_augmentations,
        mean=mean,
        std=std,
        random_erasing_p=random_erasing_p,
    )
    train_clean_dataset = EMNISTDataset(
        train_csv,
        augment=False,
        mean=mean,
        std=std,
        random_erasing_p=0.0,
    )
    val_dataset = EMNISTDataset(
        val_csv,
        augment=False,
        mean=mean,
        std=std,
        random_erasing_p=0.0,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_aug_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    train_clean_loader = DataLoader(
        train_clean_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, train_clean_loader, val_loader
