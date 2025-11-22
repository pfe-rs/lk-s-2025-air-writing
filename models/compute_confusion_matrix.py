#!/usr/bin/env python3
"""
Racuna confusion matrix za CNN model i cuva kao JSON.
Koristi se za weighted edit distance u language modelu.

Primer:
    python models/compute_confusion_matrix.py --weights "models/saved weights-labeled/best.pth"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.data import create_dataloaders, DEFAULT_NORMALIZE_MEAN, DEFAULT_NORMALIZE_STD
from models.metrics_utils import compute_confusion_matrix


class CNN(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=dropout)
        self.fcc1 = nn.Linear(128 * 3 * 3, 128)
        self.fcc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fcc1(x))
        x = self.dropout(x)
        return self.fcc2(x)


def run_inference(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds = torch.argmax(out, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--val-csv", type=Path, default=PROJECT_ROOT / "models" / "emnist-letters-test.csv")
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "dataset" / "emnist-letters-train.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "language_model" / "artifacts" / "confusion_matrix.json")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = CNN().to(device)
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Weights not found: {weights_path}")
        sys.exit(1)

    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(state)
    print(f"Loaded: {weights_path}")

    if not args.val_csv.exists():
        print(f"Val CSV not found: {args.val_csv}")
        sys.exit(1)

    if not args.train_csv.exists():
        args.train_csv = args.val_csv  # fallback

    _, _, val_loader = create_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        num_workers=4,
        mean=DEFAULT_NORMALIZE_MEAN,
        std=DEFAULT_NORMALIZE_STD,
        disable_augmentations=True,
        device=device,
    )

    print(f"Running on {len(val_loader.dataset)} samples...")
    y_true, y_pred = run_inference(model, val_loader, device)

    # confusion matrix normalizovana po redovima = P(pred|true)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=26, normalize=True)

    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) * 100
    print(f"Accuracy: {acc:.2f}%")

    # convert to dict format
    letters = [chr(ord('A') + i) for i in range(26)]
    cm_dict = {}
    for i, tl in enumerate(letters):
        cm_dict[tl] = {}
        for j, pl in enumerate(letters):
            p = float(cm[i, j])
            if p > 0.001:
                cm_dict[tl][pl] = round(p, 4)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'confusion_matrix': cm_dict,
            'accuracy': round(acc, 2),
            'num_samples': len(y_true),
            'letters': letters,
        }, f, indent=2)
    print(f"Saved: {args.output}")

    # prikazi top konfuzije
    print("\nNajcesce greske:")
    conf = []
    for i in range(26):
        for j in range(26):
            if i != j and cm[i, j] > 0.01:
                conf.append((letters[i], letters[j], cm[i, j]))
    conf.sort(key=lambda x: -x[2])
    for tl, pl, p in conf[:10]:
        print(f"  {tl} -> {pl}: {p*100:.1f}%")


if __name__ == "__main__":
    main()
