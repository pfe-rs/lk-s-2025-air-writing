from __future__ import annotations

import json
import os
import pickle
import sys
from math import log
from pathlib import Path
from typing import Any, Dict, Iterable, List

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config_utils import load_config
from lookup import SymSpellIndex, load_vocab, load_confusion_matrix, build_substitution_costs


class LanguageModel:
    def __init__(self, config_path: str = "config/language_layer.yaml", overrides: Dict[str, Any] | None = None):
        self.config = load_config(config_path)
        if overrides:
            self.config.update(overrides)
        self.language = self.config.get("language", "en")
        self.artifacts_dir = self.config.get("artifacts_dir", "models/language_model/artifacts")
        self.max_distance = int(self.config.get("max_edit_distance", 2))
        self.top_k_candidates = int(self.config.get("top_k_candidates", 5))
        self.use_bigrams = bool(self.config.get("use_bigrams", False))
        self.distance_penalty = float(self.config.get("distance_penalty", 1.0))

        vocab_path = os.path.join(self.artifacts_dir, f"vocab_{self.language}.json")
        index_path = os.path.join(self.artifacts_dir, f"symspell_index_{self.language}.pkl")
        confusion_path = os.path.join(self.artifacts_dir, "confusion_matrix.json")

        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(f"Vocabulary not found at {vocab_path}. Run build_vocab.py first.")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"Delete index not found at {index_path}. Run lookup.py first.")

        self.vocab = load_vocab(vocab_path)

        # ucitaj confusion matrix ako postoji - koristi se za weighted edit distance
        self.sub_costs = None
        cm = load_confusion_matrix(confusion_path)
        if cm:
            self.sub_costs = build_substitution_costs(cm)
            print(f"[LM] Loaded confusion matrix, using weighted edit distance")

        self.symspell = self._load_index(index_path)
        self.bigram_counts = self._load_bigram_counts() if self.use_bigrams else {}

    def _load_index(self, index_path: str) -> SymSpellIndex:
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        delete_index = data["delete_index"]
        max_distance = data.get("max_distance", self.max_distance)
        return SymSpellIndex(self.vocab, delete_index, max_distance, self.sub_costs)

    def _load_bigram_counts(self) -> Dict[str, Dict[str, int]]:
        bigram_path = os.path.join(self.artifacts_dir, f"bigrams_{self.language}.json")
        if not os.path.isfile(bigram_path):
            print("[INFO] Bigram counts not found; proceeding without them.")
            return {}
        with open(bigram_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def correct(self, word: str, history: Iterable[str] | None = None) -> str:
        if not word:
            return word

        history_list = list(history or [])
        candidate_list = list(self.symspell.candidates(word, top_k=self.top_k_candidates))
        seen = {candidate for candidate, _ in candidate_list}
        if word not in seen:
            candidate_list.append((word, 0.0))

        scored_candidates = [
            (self._score_candidate(candidate, distance, history_list), candidate, distance)
            for candidate, distance in candidate_list
        ]
        best_score, best_candidate, _ = max(scored_candidates, key=lambda item: item[0])
        return best_candidate

    def _score_candidate(self, candidate: str, distance: float, history: List[str]) -> float:
        base_score = log(self.vocab.get(candidate, 1e-9))
        base_score -= self.distance_penalty * distance
        if not self.use_bigrams or not history:
            return base_score
        previous = history[-1]
        bigram_score = self._bigram_log_prob(previous, candidate)
        return base_score + bigram_score

    def _bigram_log_prob(self, prev_word: str, current_word: str) -> float:
        prev_counts = self.bigram_counts.get(prev_word)
        if not prev_counts:
            return 0.0
        count = prev_counts.get(current_word, 0)
        total = sum(prev_counts.values())
        if count == 0:
            return log(1.0 / (total + len(prev_counts)))
        return log(count / total)


_DEFAULT_MODEL: LanguageModel | None = None


def get_default_model() -> LanguageModel:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = LanguageModel()
    return _DEFAULT_MODEL


def correct_word(word: str, history: Iterable[str] | None = None) -> str:
    model = get_default_model()
    return model.correct(word, history)
