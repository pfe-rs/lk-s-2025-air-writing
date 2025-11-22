#!/usr/bin/env python3
"""
Evaluate how distance_penalty affects language-model correction accuracy
on a synthetic dataset with injected recognition errors.

Outputs:
    - metrics CSV with accuracy per penalty value
    - dataset CSV with ground truth and corrupted words
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.language_model.config_utils import load_config
from models.language_model.language_model import LanguageModel

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def load_vocab(vocab_path: Path, limit: int | None = None) -> List[Tuple[str, float]]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        data: Dict[str, float] = json.load(f)
    items = sorted(data.items(), key=lambda item: item[1], reverse=True)
    if limit:
        items = items[:limit]
    return items


def corrupt_word(word: str, max_changes: int = 2) -> str:
    if len(word) == 0:
        return word
    operations = []
    operations.append("substitute")
    if len(word) > 1:
        operations.append("delete")
        operations.append("transpose")
    operations.append("insert")
    num_changes = random.randint(1, max_changes)
    corrupted = list(word)
    original = word
    for _ in range(10):
        corrupted = list(original)
        length = len(corrupted)
        for _ in range(num_changes):
            op = random.choice(operations)
            if op == "substitute" and length > 0:
                idx = random.randrange(len(corrupted))
                choices = ALPHABET.replace(corrupted[idx], "") if corrupted[idx] in ALPHABET else ALPHABET
                corrupted[idx] = random.choice(choices) if choices else random.choice(ALPHABET)
            elif op == "delete" and len(corrupted) > 1:
                idx = random.randrange(len(corrupted))
                del corrupted[idx]
            elif op == "insert":
                idx = random.randrange(len(corrupted) + 1)
                corrupted.insert(idx, random.choice(ALPHABET))
            elif op == "transpose" and len(corrupted) > 1:
                idx = random.randrange(len(corrupted) - 1)
                if corrupted[idx] != corrupted[idx + 1]:
                    corrupted[idx], corrupted[idx + 1] = corrupted[idx + 1], corrupted[idx]
        corrupted_word = "".join(corrupted)
        if corrupted_word != original and corrupted_word:
            return corrupted_word
    return original if original else random.choice(ALPHABET)


def build_dataset(words: List[str], seed: int = 42) -> List[Tuple[str, str]]:
    random.seed(seed)
    dataset: List[Tuple[str, str]] = []
    for word in words:
        corrupted = corrupt_word(word)
        dataset.append((word, corrupted))
    return dataset


def evaluate_penalty(dataset: List[Tuple[str, str]], penalty: float, config: Dict, history_mode: str = "none") -> Dict[str, float]:
    overrides = {"distance_penalty": penalty}
    model = LanguageModel(overrides=overrides)
    total = len(dataset)
    corrected_correct = 0
    helpful = 0
    harmful = 0
    unchanged = 0
    history: List[str] = []

    for ground_truth, raw_word in dataset:
        corrected = model.correct(raw_word, history if history_mode == "seq" else None)
        if corrected == ground_truth:
            corrected_correct += 1
            if corrected != raw_word:
                helpful += 1
        else:
            if corrected == raw_word:
                unchanged += 1
            else:
                harmful += 1
        if history_mode == "seq":
            history.append(corrected)
    return {
        "penalty": penalty,
        "total": total,
        "corrected_accuracy": corrected_correct / total,
        "helpful": helpful,
        "harmful": harmful,
        "unchanged": unchanged,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep distance_penalty and report correction accuracy.")
    parser.add_argument("--config", default="config/language_layer.yaml", help="Config file to read base settings.")
    parser.add_argument("--vocab", default=None, help="Optional path to vocab JSON.")
    parser.add_argument("--sample", type=int, default=1000, help="Number of top vocabulary words to sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--penalties", type=str, default="0,0.25,0.5,0.75,1,1.5,2,2.5,3,4", help="Comma-separated penalty values to test.")
    parser.add_argument("--dataset-output", default="language_model_dataset.csv", help="CSV for synthetic dataset.")
    parser.add_argument("--metrics-output", default="language_model_metrics.csv", help="CSV for metrics.")
    parser.add_argument("--history-mode", choices=["none", "seq"], default="none", help="Whether to feed previous corrections as history.")
    args = parser.parse_args()

    config = load_config(args.config)
    language = config.get("language", "en")
    artifacts_dir = config.get("artifacts_dir", "models/language_model/artifacts")
    vocab_path = Path(args.vocab) if args.vocab else Path(artifacts_dir) / f"vocab_{language}.json"
    vocab_items = load_vocab(vocab_path, limit=args.sample)
    words = [word for word, _ in vocab_items]
    dataset = build_dataset(words, seed=args.seed)

    dataset_csv = Path(args.dataset_output)
    with open(dataset_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ground_truth", "raw_word"])
        writer.writerows(dataset)
    print(f"Wrote synthetic dataset with {len(dataset)} rows to {dataset_csv}")

    penalties = [float(value.strip()) for value in args.penalties.split(",")]
    metrics: List[Dict[str, float]] = []
    for penalty in penalties:
        result = evaluate_penalty(dataset, penalty, config, history_mode=args.history_mode)
        metrics.append(result)
        print(
            f"penalty={penalty:.2f} | corrected_acc={result['corrected_accuracy']:.3f} | helpful={result['helpful']} harmful={result['harmful']} unchanged={result['unchanged']}"
        )

    metrics_csv = Path(args.metrics_output)
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["penalty", "corrected_accuracy", "helpful", "harmful", "unchanged", "total"])
        for row in metrics:
            writer.writerow(
                [
                    row["penalty"],
                    row["corrected_accuracy"],
                    row["helpful"],
                    row["harmful"],
                    row["unchanged"],
                    row["total"],
                ]
            )
    best = max(metrics, key=lambda r: r["corrected_accuracy"])
    print(
        f"Best corrected accuracy {best['corrected_accuracy']:.3f} "
        f"achieved at penalty={best['penalty']:.2f}"
    )


if __name__ == "__main__":
    main()
