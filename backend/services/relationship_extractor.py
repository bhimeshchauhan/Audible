"""
Relationship extraction using LLM analysis.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from openai import OpenAI
from tqdm import tqdm

from .character_extractor import CharacterExtractor
from .text_utils import CATEGORY_COLORS, get_relationship_category


class RelationshipExtractor:
    """Extracts typed relationships using LLM."""

    def __init__(
        self,
        nlp,
        character_extractor: CharacterExtractor,
        llm_client: OpenAI,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.nlp = nlp
        self.char_extractor = character_extractor
        self.client = llm_client
        self.relationships: List[Dict] = []
        self.interaction_graph = nx.Graph()

        # Configuration with defaults
        cfg = config or {}
        self.min_sentence_len = cfg.get("min_sentence_len", 10)
        self.window_sentences = cfg.get("window_sentences", 8)
        self.window_stride = cfg.get("window_stride", 4)
        self.max_context_chars = cfg.get("max_context_chars", 1600)
        self.max_people_per_window = cfg.get("max_people_per_window", 10)
        self.min_interaction_weight = cfg.get("min_interaction_weight", 2)
        self.evidence_per_pair = cfg.get("evidence_per_pair", 6)
        self.max_pairs_to_describe = cfg.get("max_pairs_to_describe", 400)
        self.min_quote_words = cfg.get("min_quote_words", 5)
        self.max_quote_words = cfg.get("max_quote_words", 40)
        self.require_quotes = cfg.get("require_quotes", True)
        self.model = cfg.get("model", "llama3.2")
        self.temperature = cfg.get("temperature", 0.0)
        self.max_workers = cfg.get("max_workers", 2)
        self.min_request_interval_sec = cfg.get("min_request_interval_sec", 0.1)
        self.max_retries = cfg.get("max_retries", 3)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        doc = self.nlp(text)
        return [
            s.text.strip()
            for s in doc.sents
            if len(s.text.strip()) >= self.min_sentence_len
        ]

    def _create_windows(self, sentences: List[str]) -> List[Tuple[int, str]]:
        """Create overlapping text windows for analysis."""
        out: List[Tuple[int, str]] = []
        i = 0
        while i < len(sentences):
            chunk = " ".join(sentences[i : i + self.window_sentences])
            chunk = re.sub(r"\s+", " ", chunk).strip()
            if chunk:
                out.append((i, chunk[: self.max_context_chars]))
            i += self.window_stride
        return out

    def _extract_evidence(self, window_text: str, a: str, b: str, k: int) -> List[str]:
        """Extract evidence sentences for a character pair."""
        doc = self.nlp(window_text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        ev: List[str] = []
        for s in sents:
            if a in s and b in s:
                ev.append(s)
        if not ev:
            for s in sents:
                if a in s or b in s:
                    ev.append(s)
                if len(ev) >= k:
                    break
        seen: Set[str] = set()
        out: List[str] = []
        for e in ev:
            if e not in seen:
                seen.add(e)
                out.append(e)
            if len(out) >= k:
                break
        return out

    def _build_interaction_graph(
        self, windows: List[Tuple[int, str]]
    ) -> Tuple[nx.Graph, Dict]:
        """Build co-occurrence interaction graph."""
        print("[INFO] Building interaction graph...")
        edges: Counter[Tuple[str, str]] = Counter()
        evidence: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for _, wtxt in tqdm(windows, desc="Co-occurrence"):
            present = self.char_extractor.get_characters_in_text(wtxt)
            if len(present) < 2:
                continue
            if len(present) > self.max_people_per_window:
                present = present[: self.max_people_per_window]

            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a, b = present[i], present[j]
                    key = (a, b) if a < b else (b, a)
                    edges[key] += 1
                    if len(evidence[key]) < self.evidence_per_pair:
                        ev = self._extract_evidence(wtxt, a, b, self.evidence_per_pair)
                        for e in ev:
                            if len(evidence[key]) >= self.evidence_per_pair:
                                break
                            if e not in evidence[key]:
                                evidence[key].append(e)

        G = nx.Graph()
        for (a, b), w in edges.items():
            if w >= self.min_interaction_weight:
                G.add_edge(a, b, weight=int(w), evidence=evidence.get((a, b), []))
        return G, evidence

    def _build_prompt(self, a: str, b: str, evidence: List[str]) -> str:
        """Build LLM prompt for relationship analysis."""
        ev_text = "\n".join([f"- {s}" for s in evidence])
        return f"""Analyze this novel excerpt. Determine the relationship between two characters.

                Character A: {a}
                Character B: {b}

                Evidence:
                {ev_text}

                Return ONLY valid JSON:
                {{"a": "{a}", "b": "{b}", "relationship_type": "specific type", "direction": "A_to_B or B_to_A or mutual", "description": "one sentence", "quotes": ["exact quote"]}}

                Types: father, mother, son, daughter, brother, sister, husband, wife, friend, enemy, rival, colleague, mentor, servant, acquaintance
                Direction: A_to_B means A is [type] OF B. Use mutual for symmetric relations."""

    def _call_llm(self, prompt: str, attempt: int = 0) -> Optional[dict]:
        """Call LLM with retry logic."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return None
        except Exception as e:
            if attempt >= self.max_retries:
                print(f"[WARN] LLM failed: {e}")
                return None
            time.sleep(min(30.0, 2**attempt))
            return self._call_llm(prompt, attempt + 1)

    def _call_llm_batch(self, prompts: List[str]) -> List[Optional[dict]]:
        """Process multiple LLM calls in parallel."""
        results: List[Optional[dict]] = [None] * len(prompts)

        def task(ix: int, p: str) -> Tuple[int, Optional[dict]]:
            time.sleep(ix * self.min_request_interval_sec)
            return ix, self._call_llm(p)

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = [ex.submit(task, i, p) for i, p in enumerate(prompts)]
            for fut in tqdm(as_completed(futs), total=len(prompts), desc="LLM calls"):
                ix, obj = fut.result()
                results[ix] = obj
        return results

    def _verify(self, obj: Any, a: str, b: str, evidence: List[str]) -> Optional[dict]:
        """Verify and validate LLM response."""
        if not isinstance(obj, dict):
            return None
        if obj.get("a") != a or obj.get("b") != b:
            return None
        rel_type = (obj.get("relationship_type") or "").strip()
        direction = (obj.get("direction") or "mutual").strip()
        desc = (obj.get("description") or "").strip()
        quotes = obj.get("quotes", [])
        if not rel_type or not desc:
            return None

        good_quotes: List[str] = []
        if self.require_quotes and isinstance(quotes, list):
            ev_join = "\n".join(evidence)
            for q in quotes:
                if isinstance(q, str):
                    q2 = q.strip()
                    wc = len(q2.split())
                    if (
                        self.min_quote_words <= wc <= self.max_quote_words
                        and q2 in ev_join
                    ):
                        good_quotes.append(q2)
            if not good_quotes:
                return None
            quotes = good_quotes[:2]

        return {
            "a": a,
            "b": b,
            "relationship_type": rel_type.lower(),
            "direction": direction,
            "description": desc,
            "quotes": quotes,
            "evidence": evidence,
        }

    def extract(self, text: str) -> List[Dict]:
        """
        Extract relationships from text.

        Args:
            text: Full text to analyze

        Returns:
            List of relationship dictionaries
        """
        print("\n" + "=" * 60)
        print("MODULE 2: RELATIONSHIP EXTRACTION")
        print("=" * 60)

        sentences = self._split_sentences(text)
        windows = self._create_windows(sentences)
        print(f"[INFO] Created {len(windows)} text windows")

        self.interaction_graph, evidence = self._build_interaction_graph(windows)

        if self.interaction_graph.number_of_edges() == 0:
            print("[ERROR] No character interactions found!")
            return []

        print(
            f"[INFO] Found {self.interaction_graph.number_of_edges()} character pairs"
        )

        candidates = sorted(
            [
                (u, v, d["weight"], evidence.get((u, v) if u < v else (v, u), []))
                for u, v, d in self.interaction_graph.edges(data=True)
            ],
            key=lambda x: x[2],
            reverse=True,
        )[: self.max_pairs_to_describe]

        prompts: List[str] = []
        meta: List[Tuple[str, str, int, List[str]]] = []
        for a, b, w, ev in candidates:
            if ev:
                prompts.append(self._build_prompt(a, b, ev))
                meta.append((a, b, w, ev))

        print(f"[INFO] Querying LLM for {len(prompts)} character pairs...")
        responses = self._call_llm_batch(prompts)

        self.relationships = []
        for resp, (a, b, w, ev) in zip(responses, meta):
            verified = self._verify(resp, a, b, ev)
            if verified:
                verified["interaction_weight"] = w
                self.relationships.append(verified)

        print(f"\n[RESULT] Extracted {len(self.relationships)} relationships")
        type_counts: Counter[str] = Counter()
        for r in self.relationships:
            type_counts[r["relationship_type"]] += 1
        print("-" * 40)
        for rtype, count in type_counts.most_common():
            print(f"  {rtype}: {count} ({get_relationship_category(rtype)})")

        return self.relationships

    def create_graph(self) -> nx.DiGraph:
        """Create directed graph from extracted relationships."""
        G = nx.DiGraph()
        for r in self.relationships:
            a, b = r["a"], r["b"]
            rel_type = r["relationship_type"]
            direction = r.get("direction", "mutual")
            weight = r.get("interaction_weight", 1)
            category = get_relationship_category(rel_type)
            color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["unknown"])
            tooltip = f"{rel_type.title()}\n{r['description']}"

            if direction == "A_to_B":
                G.add_edge(
                    a,
                    b,
                    relationship=rel_type,
                    weight=weight,
                    color=color,
                    title=tooltip,
                )
            elif direction == "B_to_A":
                G.add_edge(
                    b,
                    a,
                    relationship=rel_type,
                    weight=weight,
                    color=color,
                    title=tooltip,
                )
            else:
                G.add_edge(
                    a,
                    b,
                    relationship=rel_type,
                    weight=weight,
                    color=color,
                    title=tooltip,
                )
        return G
