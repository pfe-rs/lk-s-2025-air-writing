#!/usr/bin/env python3
"""
Script for building a frequency vocabulary using the wordfreq package.

Outputs:
    - vocab.json: mapping word -> frequency score
    - optional bigram counts (future extension)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

from wordfreq import top_n_list, zipf_frequency

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config_utils import load_config


def build_vocab(language: str, min_zipf: float, limit: int | None = None) -> Dict[str, float]:
    words = top_n_list(language, n=limit or 1000000)
    vocab = {}
    for word in words:
        freq = zipf_frequency(word, language)
        if freq >= min_zipf:
            vocab[word] = freq
    return vocab


def main():
    parser = argparse.ArgumentParser(description="Build vocabulary using wordfreq.")
    parser.add_argument(
        "--config",
        default="config/language_layer.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of words from top_n_list.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    language = config.get("language", "en")
    min_zipf = float(config.get("min_frequency_zipf", 3.0))
    artifacts_dir = config.get("artifacts_dir", "models/language_model/artifacts")

    os.makedirs(artifacts_dir, exist_ok=True)

    vocab = build_vocab(language, min_zipf, args.limit)
    vocab_path = os.path.join(artifacts_dir, f"vocab_{language}.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary with {len(vocab)} words to {vocab_path}")


if __name__ == "__main__":
    main()
