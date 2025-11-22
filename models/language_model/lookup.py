#!/usr/bin/env python3
"""
Utilities for building and querying a SymSpell-like lookup index.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config_utils import load_config


def load_vocab(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_confusion_matrix(path: str) -> Optional[Dict[str, Dict[str, float]]]:
    """Ucitaj confusion matrix ako postoji."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("confusion_matrix")


def build_substitution_costs(confusion: Dict[str, Dict[str, float]], default_cost: float = 1.0) -> Dict[str, Dict[str, float]]:
    """Pretvori confusion matrix u substitution costs za weighted levenshtein."""
    costs: Dict[str, Dict[str, float]] = {}
    all_costs = []

    for true_char, preds in confusion.items():
        costs[true_char.lower()] = {}
        for pred_char, prob in preds.items():
            if true_char == pred_char:
                continue
            # -log daje visoku vrednost za male verovatnoce
            raw_cost = -math.log(prob + 1e-6)
            costs[true_char.lower()][pred_char.lower()] = raw_cost
            all_costs.append(raw_cost)

    if not all_costs:
        return {}

    # normalizuj na opseg [0.1, 1.5]
    min_c, max_c = min(all_costs), max(all_costs)
    if max_c - min_c < 0.001:
        return {}

    for tc in costs:
        for pc in costs[tc]:
            normalized = 0.1 + 1.4 * (costs[tc][pc] - min_c) / (max_c - min_c)
            costs[tc][pc] = round(normalized, 3)

    return costs


def generate_deletes(word: str, max_distance: int) -> Set[str]:
    deletes: Set[str] = set()

    def _helper(current: str, depth: int):
        if depth >= max_distance:
            return
        for i in range(len(current)):
            candidate = current[:i] + current[i + 1 :]
            if not candidate or candidate in deletes:
                continue
            deletes.add(candidate)
            _helper(candidate, depth + 1)

    _helper(word, 0)
    return deletes


def build_delete_index(vocab: Iterable[str], max_distance: int) -> Dict[str, List[str]]:
    index: Dict[str, Set[str]] = defaultdict(set)
    for word in vocab:
        for delete in generate_deletes(word, max_distance):
            index[delete].add(word)
    # Convert to lists for serialization
    return {key: sorted(values) for key, values in index.items()}


class SymSpellIndex:
    def __init__(
        self,
        vocab: Dict[str, float],
        delete_index: Dict[str, List[str]],
        max_distance: int,
        sub_costs: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.vocab = vocab
        self.delete_index = delete_index
        self.max_distance = max_distance
        self.sub_costs = sub_costs  # None = koristi standard levenshtein

    def _compute_distance(self, query: str, candidate: str) -> float:
        if self.sub_costs:
            return weighted_levenshtein(query, candidate, self.sub_costs)
        return float(bounded_levenshtein(query, candidate, self.max_distance))

    def candidates(self, query: str, top_k: int | None = None) -> List[Tuple[str, float]]:
        suggestions: Dict[str, float] = {}
        queue = {query}
        visited = set()

        def consider(candidate: str):
            distance = self._compute_distance(query, candidate)
            if distance <= self.max_distance:
                prev = suggestions.get(candidate)
                if prev is None or distance < prev:
                    suggestions[candidate] = distance

        while queue:
            term = queue.pop()
            if term in visited:
                continue
            visited.add(term)

            if term in self.vocab:
                consider(term)

            if len(query) - len(term) < self.max_distance:
                for i in range(len(term)):
                    deletion = term[:i] + term[i + 1 :]
                    if deletion and deletion not in visited:
                        queue.add(deletion)

            for suggestion in self.delete_index.get(term, []):
                consider(suggestion)

        sorted_suggestions = sorted(
            suggestions.items(),
            key=lambda item: (item[1], -self.vocab.get(item[0], 0.0)),
        )
        if top_k is not None:
            return sorted_suggestions[:top_k]
        return sorted_suggestions


def bounded_levenshtein(source: str, target: str, max_distance: int) -> int:
    if source == target:
        return 0
    len_src, len_tgt = len(source), len(target)
    if abs(len_src - len_tgt) > max_distance:
        return max_distance + 1
    if len_src == 0:
        return len_tgt if len_tgt <= max_distance else max_distance + 1
    if len_tgt == 0:
        return len_src if len_src <= max_distance else max_distance + 1

    if len_src > len_tgt:
        source, target = target, source
        len_src, len_tgt = len_tgt, len_src

    previous_row = list(range(len_tgt + 1))
    for i, src_char in enumerate(source, start=1):
        current_row = [i]
        min_current = current_row[0]
        for j, tgt_char in enumerate(target, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (src_char != tgt_char)
            cost = min(insert_cost, delete_cost, replace_cost)
            current_row.append(cost)
            if cost < min_current:
                min_current = cost
        if min_current > max_distance:
            return max_distance + 1
        previous_row = current_row

    distance = previous_row[-1]
    return distance if distance <= max_distance else max_distance + 1


def weighted_levenshtein(
    source: str,
    target: str,
    sub_costs: Dict[str, Dict[str, float]],
    default_sub: float = 1.0,
    insert_del_cost: float = 1.0,
) -> float:
    """
    Levenshtein sa custom substitution costs.
    sub_costs[a][b] = cena zamene a sa b
    """
    if source == target:
        return 0.0

    len_src, len_tgt = len(source), len(target)
    if len_src == 0:
        return len_tgt * insert_del_cost
    if len_tgt == 0:
        return len_src * insert_del_cost

    # dp matrica
    prev = [j * insert_del_cost for j in range(len_tgt + 1)]

    for i, sc in enumerate(source, start=1):
        curr = [i * insert_del_cost]
        for j, tc in enumerate(target, start=1):
            if sc == tc:
                curr.append(prev[j - 1])
            else:
                # cena zamene iz confusion matrix
                s_cost = sub_costs.get(sc, {}).get(tc, default_sub)
                replace = prev[j - 1] + s_cost
                insert = curr[j - 1] + insert_del_cost
                delete = prev[j] + insert_del_cost
                curr.append(min(replace, insert, delete))
        prev = curr

    return prev[-1]


def main():
    parser = argparse.ArgumentParser(description="Build SymSpell-like index from vocabulary.")
    parser.add_argument("--config", default="config/language_layer.yaml")
    parser.add_argument("--vocab", default=None, help="Path to vocabulary JSON. Defaults to artifacts/vocab_LANG.json")
    parser.add_argument("--max_distance", type=int, default=None, help="Override max edit distance.")
    args = parser.parse_args()

    config = load_config(args.config)
    language = config.get("language", "en")
    artifacts_dir = config.get("artifacts_dir", "models/language_model/artifacts")
    vocab_path = args.vocab or os.path.join(artifacts_dir, f"vocab_{language}.json")
    max_distance = args.max_distance if args.max_distance is not None else int(config.get("max_edit_distance", 2))

    vocab = load_vocab(vocab_path)
    delete_index = build_delete_index(vocab.keys(), max_distance)
    index_path = os.path.join(artifacts_dir, f"symspell_index_{language}.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(
            {
                "max_distance": max_distance,
                "delete_index": delete_index,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved delete index with {len(delete_index)} keys to {index_path}")


if __name__ == "__main__":
    main()
