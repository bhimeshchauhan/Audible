"""
Character extraction using NER + coreference resolution.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm

from .text_utils import normalize_name


class CharacterExtractor:
    """Extracts characters using NER + coreference resolution."""

    def __init__(self, nlp, min_person_freq: int = 2):
        self.nlp = nlp
        self.min_person_freq = min_person_freq
        self.characters: Set[str] = set()
        self.alias_map: Dict[str, str] = {}
        self.character_freq: Counter[str] = Counter()

    def _build_alias_map(self, freq: Counter[str]) -> Dict[str, str]:
        """Build mapping from short names to full names."""
        full = [n for n in freq if len(n.split()) >= 2]
        by_last: Dict[str, List[str]] = defaultdict(list)
        for fn in full:
            by_last[fn.split()[-1]].append(fn)
        alias: Dict[str, str] = {}
        for last, cands in by_last.items():
            cands = sorted(cands, key=lambda x: freq[x], reverse=True)
            if not cands:
                continue
            top = cands[0]
            second = cands[1] if len(cands) > 1 else None
            if second is None or freq[top] >= 2 * max(1, freq[second]):
                alias[last] = top
        return alias

    def _apply_alias(self, name: str) -> str:
        """Apply alias mapping to normalize name references."""
        parts = name.split()
        if len(parts) == 1 and parts[0] in self.alias_map:
            return self.alias_map[parts[0]]
        return name

    def extract(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Extract characters from sentences.

        Args:
            sentences: List of sentence strings

        Returns:
            Dictionary with characters, alias_map, and frequency data
        """
        print("\n" + "=" * 60)
        print("MODULE 1: CHARACTER EXTRACTION")
        print("=" * 60)
        print("[INFO] Discovering characters using NER...")

        freq: Counter[str] = Counter()
        for doc in tqdm(
            self.nlp.pipe(sentences, batch_size=64, disable=["fastcoref"]),
            total=len(sentences),
            desc="NER scan",
        ):
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    nm = normalize_name(ent.text)
                    if nm:
                        freq[nm] += 1

        self.alias_map = self._build_alias_map(freq)

        merged: Counter[str] = Counter()
        for nm, c in freq.items():
            merged[self._apply_alias(nm)] += c

        self.characters = {n for n, c in merged.items() if c >= self.min_person_freq}
        self.character_freq = Counter(
            {n: c for n, c in merged.items() if n in self.characters}
        )

        print(
            f"\n[RESULT] Found {len(self.characters)} characters (freq >= {self.min_person_freq}):"
        )
        print("-" * 40)
        for name, count in self.character_freq.most_common(30):
            print(f"  {name}: {count} mentions")

        return {
            "characters": self.characters,
            "alias_map": self.alias_map,
            "frequency": dict(self.character_freq),
        }

    def get_characters_in_text(self, text: str) -> List[str]:
        """Find all known characters mentioned in a text."""
        doc = self.nlp(text)
        found: List[str] = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                nm = self._apply_alias(normalize_name(ent.text))
                if nm in self.characters:
                    found.append(nm)

        cmap = self._coref_map(doc)
        for rep in cmap.values():
            nm = self._apply_alias(normalize_name(rep))
            if nm in self.characters:
                found.append(nm)

        seen: Set[str] = set()
        uniq: List[str] = []
        for x in found:
            if x and x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _coref_map(self, doc) -> Dict[Tuple[int, int], str]:
        """Extract coreference mapping from doc."""
        mapping: Dict[Tuple[int, int], str] = {}
        clusters = getattr(doc._, "coref_clusters", None)
        if not clusters:
            return mapping
        if hasattr(clusters[0], "mentions"):
            for cl in clusters:
                rep = cl.main.text if getattr(cl, "main", None) else cl.mentions[0].text
                for m in cl.mentions:
                    mapping[(m.start_char, m.end_char)] = rep
        return mapping
