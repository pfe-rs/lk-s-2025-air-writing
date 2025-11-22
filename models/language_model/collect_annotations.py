#!/usr/bin/env python3
"""
Interactively build a CSV dataset with ground-truth words, raw CNN outputs,
and language-model corrections for saved air-writing samples.

Usage:
    python models/language_model/collect_annotations.py \
        --input-dir Writing_part/data \
        --output annotations.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.language_model.language_model import LanguageModel


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.fcc1 = nn.Linear(128 * 3 * 3, 128)
        self.fcc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fcc1(x))
        x = self.dropout(x)
        x = self.fcc2(x)
        return x


def load_emnist_mapping(mapping_path: Path) -> dict[int, str]:
    mapping = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                idx = int(parts[0])
                letter = chr(int(parts[1]))
                mapping[idx] = letter
    return mapping


def resize_with_padding(img: np.ndarray, size: int = 28, margin: int = 2) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=img.dtype)
    max_dim = max(h, w)
    scale = (size - 2 * margin) / max_dim if max_dim > 0 else 1.0
    if scale <= 0:
        scale = 1.0
    scale = min(scale, 3.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    canvas = np.zeros((size, size), dtype=img.dtype)
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def segment_letters(word_img: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 15))
    morphed = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    boxes.sort(key=lambda b: b[0])
    img_h, img_w = word_img.shape[:2]
    min_w, min_h = 10, 10
    max_w, max_h = int(0.9 * img_w), int(0.9 * img_h)
    letter_images: List[np.ndarray] = []
    for (x, y, w, h) in boxes:
        if w < min_w or h < min_h:
            continue
        if w > max_w and h > max_h:
            continue
        pad = 5
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, img_w)
        y2 = min(y + h + pad, img_h)
        letter_img = word_img[y1:y2, x1:x2]
        letter_gray = cv2.cvtColor(letter_img, cv2.COLOR_BGR2GRAY)
        _, letter_bin = cv2.threshold(letter_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        letter_resized = resize_with_padding(letter_bin, size=28, margin=2)
        letter_images.append(letter_resized)
    return letter_images


def recognize_word(word_img: np.ndarray, model: CNN, device: torch.device, labels_map: dict[int, str]) -> Tuple[str, List[np.ndarray]]:
    letter_images = segment_letters(word_img)
    recognized_letters: List[str] = []
    if not letter_images:
        return "", []
    for letter in letter_images:
        tensor = torch.tensor(letter, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        tensor = tensor.to(device)
        with torch.no_grad():
            output = model(tensor)
            pred_idx = int(torch.argmax(output, dim=1).item())
            emnist_label = pred_idx + 1
            recognized_letters.append(labels_map.get(emnist_label, "?"))
    word = "".join(recognized_letters)
    return word, letter_images


def main():
    parser = argparse.ArgumentParser(description="Collect ground-truth annotations for saved word images.")
    parser.add_argument("--input-dir", default="Writing_part/data", help="Directory with saved word images.")
    parser.add_argument("--output", default="annotations.csv", help="CSV file to write annotations into.")
    parser.add_argument("--append", action="store_true", help="Append to CSV instead of overwriting.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of images.")
    parser.add_argument("--show", action="store_true", help="Show each image in a window while annotating.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    weights_dir = REPO_ROOT / "models" / "saved weights-labeled"
    if not weights_dir.exists():
        raise FileNotFoundError(f"Model weights directory not found: {weights_dir}")
    candidates = sorted(weights_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No .pth files found in {weights_dir}")
    model_path = candidates[-1]
    mapping_path = REPO_ROOT / "models" / "emnist-letters-mapping.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = CNN().to(device)
    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    cnn_model.eval()

    labels_map = load_emnist_mapping(mapping_path)
    language_model = LanguageModel()

    images = sorted([p for p in input_dir.glob("*.png")])
    if args.limit:
        images = images[: args.limit]

    csv_mode = "a" if args.append and Path(args.output).exists() else "w"
    with open(args.output, csv_mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if csv_mode == "w":
            writer.writerow(["filename", "ground_truth", "raw_word", "corrected_word"])

        for idx, image_path in enumerate(images, start=1):
            word_img = cv2.imread(str(image_path))
            if word_img is None:
                print(f"[WARN] Cannot read image: {image_path}")
                continue
            raw_word, letter_imgs = recognize_word(word_img, cnn_model, device, labels_map)
            corrected_word = language_model.correct(raw_word.lower()) if raw_word else ""
            if args.show:
                display = word_img.copy()
                stacked = None
                if letter_imgs:
                    resized_letters = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (56, 56), interpolation=cv2.INTER_NEAREST) for img in letter_imgs]
                    stacked = np.hstack(resized_letters)
                cv2.imshow("Word", display)
                if stacked is not None:
                    cv2.imshow("Letters", stacked)
            print(f"[{idx}/{len(images)}] {image_path.name} -> raw='{raw_word}' corrected='{corrected_word}'")
            ground_truth = input(" Ground truth (enter to skip): ").strip()
            if args.show:
                cv2.destroyAllWindows()
            if not ground_truth:
                continue
            writer.writerow([image_path.name, ground_truth, raw_word, corrected_word])
            print(" Saved.\n")


if __name__ == "__main__":
    main()
