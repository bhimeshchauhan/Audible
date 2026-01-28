#!/usr/bin/env python3
"""
Character Relationship Extraction from Book PDFs

Two-Module Architecture:
  Module 1: Character Extraction - discovers all characters using NER + coref
  Module 2: Relationship Extraction - establishes typed relationships via LLM
             and presents them in an interactive graph

Supports both LOCAL (Ollama) and CLOUD (OpenAI) LLMs.
Default: Local mode with Ollama (free, no API key needed)
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import time
import argparse
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

import PyPDF2
import spacy
import networkx as nx
from tqdm import tqdm
from pyvis.network import Network
from fastcoref import spacy_component  # registers "fastcoref"

from openai import OpenAI


# =========================
# Configuration
# =========================


@dataclass
class Config:
    # PDF
    max_pages: Optional[int] = None

    # spaCy
    spacy_model: str = "en_core_web_sm"
    enable_coref: bool = True
    coref_model_arch: str = "FCoref"

    # Segmentation
    min_sentence_len: int = 10
    window_sentences: int = 8
    window_stride: int = 4
    max_context_chars: int = 1600

    # Characters
    min_person_freq: int = 2
    max_people_per_window: int = 10

    # Candidate selection
    min_interaction_weight: int = 2
    evidence_per_pair: int = 6
    max_pairs_to_describe: int = 400

    # LLM Configuration
    use_local: bool = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
    local_model: str = os.getenv("LOCAL_MODEL", "llama3.2")
    local_base_url: str = os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm_temperature: float = 0.0

    # Verification
    min_quote_words: int = 5
    max_quote_words: int = 40
    require_quotes: bool = True

    # Outputs
    out_dir: str = "results"

    # Concurrency
    max_workers: int = int(os.getenv("LLM_MAX_WORKERS", "2"))
    min_request_interval_sec: float = float(
        os.getenv("LLM_MIN_REQUEST_INTERVAL_SEC", "0.1")
    )
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))


cfg = Config()


# =========================
# Utilities
# =========================

SURROGATE_RE = re.compile(r"[\uD800-\uDFFF]")
POSSESSIVE_RE = re.compile(r"('s|'s)$", re.I)
HONORIFICS = {
    "mr",
    "mrs",
    "ms",
    "miss",
    "dr",
    "sir",
    "madam",
    "lady",
    "lord",
    "capt",
    "captain",
    "rev",
    "fr",
    "st",
    "saint",
    "ser",
}

RELATIONSHIP_CATEGORIES = {
    "family": [
        "father",
        "mother",
        "son",
        "daughter",
        "brother",
        "sister",
        "sibling",
        "grandfather",
        "grandmother",
        "grandson",
        "granddaughter",
        "uncle",
        "aunt",
        "nephew",
        "niece",
        "cousin",
        "parent",
        "child",
    ],
    "romantic": [
        "spouse",
        "husband",
        "wife",
        "lover",
        "fiancé",
        "fiancée",
        "partner",
        "romantic interest",
        "admirer",
        "suitor",
        "betrothed",
    ],
    "friendship": ["friend", "best friend", "companion", "confidant", "ally"],
    "professional": [
        "colleague",
        "employer",
        "employee",
        "boss",
        "mentor",
        "student",
        "teacher",
        "master",
        "apprentice",
        "doctor",
        "servant",
    ],
    "social": ["neighbor", "acquaintance", "maid", "butler", "governess"],
    "antagonistic": ["enemy", "rival", "adversary", "nemesis", "opponent"],
}

CATEGORY_COLORS = {
    "family": "#e74c3c",
    "romantic": "#e91e63",
    "friendship": "#3498db",
    "professional": "#9b59b6",
    "social": "#27ae60",
    "antagonistic": "#f39c12",
    "unknown": "#95a5a6",
}


def out_path(filename: str) -> str:
    return os.path.join(cfg.out_dir, filename)


def ensure_results_dir() -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)


def sanitize_unicode(text: str) -> str:
    if not text:
        return ""
    text = SURROGATE_RE.sub("", text)
    text = "".join(
        ch for ch in text if ch in ("\n", "\t") or (0x20 <= ord(ch) <= 0x10FFFF)
    )
    return text.encode("utf-8", "replace").decode("utf-8", "replace")


def clean_text(s: str) -> str:
    s = sanitize_unicode(s)
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def log_score(x: int) -> float:
    return 1.0 + math.log(1 + x)


def get_relationship_category(rel_type: str) -> str:
    rel_lower = rel_type.lower().strip()
    for category, types in RELATIONSHIP_CATEGORIES.items():
        for t in types:
            if t in rel_lower or rel_lower in t:
                return category
    return "unknown"


# =========================
# PDF Extraction
# =========================


def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int]) -> str:
    print("[INFO] Extracting text from PDF...")
    out: List[str] = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages = reader.pages if max_pages is None else reader.pages[:max_pages]
        for page in pages:
            t = page.extract_text() or ""
            if t.strip():
                out.append(t)
    return clean_text("\n".join(out))


# ==============================================================================
# MODULE 1: CHARACTER EXTRACTION
# ==============================================================================


class CharacterExtractor:
    """Extracts characters using NER + coreference resolution."""

    def __init__(self, nlp):
        self.nlp = nlp
        self.characters: Set[str] = set()
        self.alias_map: Dict[str, str] = {}
        self.character_freq: Counter[str] = Counter()

    def _normalize_name(self, name: str) -> str:
        name = sanitize_unicode(name).strip()
        name = POSSESSIVE_RE.sub("", name)
        name = re.sub(r"^[\"'" "'']+|[\"'" "'']+$", "", name)
        name = re.sub(r"[^A-Za-z.\- '']+", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        parts = name.split()
        if parts and parts[0].rstrip(".").lower() in HONORIFICS:
            parts = parts[1:]
        name = " ".join(parts).strip()
        if len(name) < 2:
            return ""
        return " ".join([p[:1].upper() + p[1:] for p in name.split()])

    def _build_alias_map(self, freq: Counter[str]) -> Dict[str, str]:
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
        parts = name.split()
        if len(parts) == 1 and parts[0] in self.alias_map:
            return self.alias_map[parts[0]]
        return name

    def extract(self, sentences: List[str]) -> Dict[str, Any]:
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
                    nm = self._normalize_name(ent.text)
                    if nm:
                        freq[nm] += 1

        self.alias_map = self._build_alias_map(freq)

        merged: Counter[str] = Counter()
        for nm, c in freq.items():
            merged[self._apply_alias(nm)] += c

        self.characters = {n for n, c in merged.items() if c >= cfg.min_person_freq}
        self.character_freq = Counter(
            {n: c for n, c in merged.items() if n in self.characters}
        )

        print(
            f"\n[RESULT] Found {len(self.characters)} characters (freq >= {cfg.min_person_freq}):"
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
        doc = self.nlp(text)
        found: List[str] = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                nm = self._apply_alias(self._normalize_name(ent.text))
                if nm in self.characters:
                    found.append(nm)

        cmap = self._coref_map(doc)
        for rep in cmap.values():
            nm = self._apply_alias(self._normalize_name(rep))
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


# ==============================================================================
# MODULE 2: RELATIONSHIP EXTRACTION
# ==============================================================================


class RelationshipExtractor:
    """Extracts typed relationships using LLM."""

    def __init__(
        self, nlp, character_extractor: CharacterExtractor, llm_client: OpenAI
    ):
        self.nlp = nlp
        self.char_extractor = character_extractor
        self.client = llm_client
        self.relationships: List[Dict] = []
        self.interaction_graph = nx.Graph()

    def _split_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [
            s.text.strip()
            for s in doc.sents
            if len(s.text.strip()) >= cfg.min_sentence_len
        ]

    def _create_windows(self, sentences: List[str]) -> List[Tuple[int, str]]:
        out: List[Tuple[int, str]] = []
        i = 0
        while i < len(sentences):
            chunk = " ".join(sentences[i : i + cfg.window_sentences])
            chunk = re.sub(r"\s+", " ", chunk).strip()
            if chunk:
                out.append((i, chunk[: cfg.max_context_chars]))
            i += cfg.window_stride
        return out

    def _extract_evidence(self, window_text: str, a: str, b: str, k: int) -> List[str]:
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
        print("[INFO] Building interaction graph...")
        edges: Counter[Tuple[str, str]] = Counter()
        evidence: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for _, wtxt in tqdm(windows, desc="Co-occurrence"):
            present = self.char_extractor.get_characters_in_text(wtxt)
            if len(present) < 2:
                continue
            if len(present) > cfg.max_people_per_window:
                present = present[: cfg.max_people_per_window]

            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a, b = present[i], present[j]
                    key = (a, b) if a < b else (b, a)
                    edges[key] += 1
                    if len(evidence[key]) < cfg.evidence_per_pair:
                        ev = self._extract_evidence(wtxt, a, b, cfg.evidence_per_pair)
                        for e in ev:
                            if len(evidence[key]) >= cfg.evidence_per_pair:
                                break
                            if e not in evidence[key]:
                                evidence[key].append(e)

        G = nx.Graph()
        for (a, b), w in edges.items():
            if w >= cfg.min_interaction_weight:
                G.add_edge(a, b, weight=int(w), evidence=evidence.get((a, b), []))
        return G, evidence

    def _build_prompt(self, a: str, b: str, evidence: List[str]) -> str:
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
        model = cfg.local_model if cfg.use_local else cfg.openai_model
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.llm_temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return None
        except Exception as e:
            if attempt >= cfg.max_retries:
                print(f"[WARN] LLM failed: {e}")
                return None
            time.sleep(min(30.0, 2**attempt))
            return self._call_llm(prompt, attempt + 1)

    def _call_llm_batch(self, prompts: List[str]) -> List[Optional[dict]]:
        results: List[Optional[dict]] = [None] * len(prompts)

        def task(ix: int, p: str) -> Tuple[int, Optional[dict]]:
            time.sleep(ix * cfg.min_request_interval_sec)
            return ix, self._call_llm(p)

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            futs = [ex.submit(task, i, p) for i, p in enumerate(prompts)]
            for fut in tqdm(as_completed(futs), total=len(prompts), desc="LLM calls"):
                ix, obj = fut.result()
                results[ix] = obj
        return results

    def _verify(self, obj: Any, a: str, b: str, evidence: List[str]) -> Optional[dict]:
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
        if cfg.require_quotes and isinstance(quotes, list):
            ev_join = "\n".join(evidence)
            for q in quotes:
                if isinstance(q, str):
                    q2 = q.strip()
                    wc = len(q2.split())
                    if (
                        cfg.min_quote_words <= wc <= cfg.max_quote_words
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
        )[: cfg.max_pairs_to_describe]

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


# ==============================================================================
# VISUALIZATION
# ==============================================================================


def write_relationship_graph(
    G: nx.DiGraph, out_html: str, title: str = "Character Relationships"
):
    node_weights: Counter[str] = Counter()
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        node_weights[u] += w
        node_weights[v] += w

    nodes_data = [
        {"id": n, "label": n, "value": 10 + 4 * log_score(node_weights.get(n, 1))}
        for n in G.nodes()
    ]

    edges_data = []
    seen = set()
    for u, v, d in G.edges(data=True):
        key = (min(u, v), max(u, v), d.get("relationship", ""))
        if key in seen:
            continue
        seen.add(key)
        edges_data.append(
            {
                "from": u,
                "to": v,
                "label": d.get("relationship", ""),
                "title": d.get("title", ""),
                "color": {"color": d.get("color", "#95a5a6")},
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
            }
        )

    legend = [
        f'<span style="color:{c}">●</span> {cat.title()}'
        for cat, c in CATEGORY_COLORS.items()
        if cat != "unknown"
    ]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
<style>
body {{ font-family: Arial; background: linear-gradient(135deg, #1a1a2e, #16213e); min-height: 100vh; margin: 0; color: #ecf0f1; }}
.header {{ text-align: center; padding: 20px; background: rgba(0,0,0,0.3); }}
.header h1 {{ font-size: 2rem; margin: 0; }}
.legend {{ display: flex; justify-content: center; gap: 15px; padding: 10px; background: rgba(0,0,0,0.2); font-size: 0.85rem; }}
#network {{ width: 100%; height: calc(100vh - 120px); }}
</style></head><body>
<div class="header"><h1>📚 {title}</h1></div>
<div class="legend">{" | ".join(legend)}</div>
<div id="network"></div>
<script>
var nodes = new vis.DataSet({json.dumps(nodes_data)});
var edges = new vis.DataSet({json.dumps(edges_data)});
var network = new vis.Network(document.getElementById('network'), {{nodes: nodes, edges: edges}}, {{
  nodes: {{ shape: 'dot', color: {{ background: '#3498db', border: '#2980b9' }} }},
  edges: {{ width: 2 }},
  physics: {{ barnesHut: {{ gravitationalConstant: -8000, springLength: 200 }} }}
}});
</script></body></html>"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[DONE] Saved: {out_html}")


def write_interaction_graph(G: nx.Graph, out_html: str):
    net = Network(height="850px", width="100%", bgcolor="#1a1a2e", font_color="#ecf0f1")
    net.barnes_hut()
    node_w: Counter[str] = Counter()
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        node_w[u] += w
        node_w[v] += w
    for n, w in node_w.items():
        net.add_node(n, label=n, value=8 + 3 * log_score(w), color="#3498db")
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        net.add_edge(u, v, value=1 + log_score(w), label=str(w), color="#7f8c8d")
    net.write_html(out_html)
    print(f"[DONE] Saved: {out_html}")


# ==============================================================================
# MAIN
# ==============================================================================


def build_nlp():
    nlp = spacy.load(cfg.spacy_model)
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)
    if cfg.enable_coref and "fastcoref" not in nlp.pipe_names:
        nlp.add_pipe("fastcoref", config={"model_architecture": cfg.coref_model_arch})
    return nlp


def main(pdf_path: str):
    ensure_results_dir()

    out_chars = out_path("characters.json")
    out_rels = out_path("relationships.jsonl")
    out_interact = out_path("interaction_graph.html")
    out_rel_graph = out_path("character_graph.html")

    print("\n" + "=" * 60)
    print("CHARACTER RELATIONSHIP EXTRACTION")
    print("=" * 60)
    print(f"Input: {pdf_path}")
    print(f"Output: {cfg.out_dir}/")

    nlp = build_nlp()

    if cfg.use_local:
        print(f"[INFO] Using LOCAL LLM: {cfg.local_model} via Ollama")
        client = OpenAI(base_url=cfg.local_base_url, api_key="ollama")
    else:
        print(f"[INFO] Using CLOUD LLM: {cfg.openai_model} via OpenAI")
        client = OpenAI()

    text = extract_text_from_pdf(pdf_path, cfg.max_pages)

    doc = nlp(text)
    sentences = [
        s.text.strip() for s in doc.sents if len(s.text.strip()) >= cfg.min_sentence_len
    ]
    print(f"[INFO] Sentences: {len(sentences)}")

    # Module 1
    char_ext = CharacterExtractor(nlp)
    char_result = char_ext.extract(sentences)

    with open(out_chars, "w", encoding="utf-8") as f:
        json.dump(
            {
                "characters": list(char_result["characters"]),
                "frequency": char_result["frequency"],
                "alias_map": char_result["alias_map"],
            },
            f,
            indent=2,
        )
    print(f"[DONE] Saved: {out_chars}")

    if len(char_result["characters"]) < 2:
        print("[ERROR] Need at least 2 characters.")
        sys.exit(2)

    # Module 2
    rel_ext = RelationshipExtractor(nlp, char_ext, client)
    relationships = rel_ext.extract(text)

    if not relationships:
        print("[ERROR] No relationships extracted.")
        sys.exit(3)

    with open(out_rels, "w", encoding="utf-8") as f:
        for r in relationships:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE] Saved: {out_rels}")

    print("\n[INFO] Generating graphs...")
    write_interaction_graph(rel_ext.interaction_graph, out_interact)
    rel_graph = rel_ext.create_graph()
    write_relationship_graph(rel_graph, out_rel_graph)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Characters: {len(char_result['characters'])}")
    print(f"  Relationships: {len(relationships)}")
    print(f"\nOpen {out_rel_graph} in a browser!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extract character relationships from a novel PDF",
        epilog="""
Examples:
  # Local mode (default, FREE - uses Ollama)
  python ner.py books/novel.pdf
  python ner.py books/novel.pdf --local_model mistral

  # Cloud mode (requires OPENAI_API_KEY)
  python ner.py books/novel.pdf --cloud

Setup for local:
  1. Install Ollama: brew install ollama
  2. Start: brew services start ollama
  3. Pull model: ollama pull llama3.2
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("pdf", help="Path to PDF")
    ap.add_argument(
        "--local", action="store_true", default=True, help="Use local Ollama (default)"
    )
    ap.add_argument("--cloud", action="store_true", help="Use OpenAI cloud API")
    ap.add_argument(
        "--local_model",
        default=cfg.local_model,
        help="Ollama model (default: llama3.2)",
    )
    ap.add_argument(
        "--model", default=cfg.openai_model, help="OpenAI model (for --cloud)"
    )
    ap.add_argument("--min_person_freq", type=int, default=cfg.min_person_freq)
    ap.add_argument("--max_pages", type=int, default=None)
    args = ap.parse_args()

    cfg.use_local = not args.cloud
    cfg.local_model = args.local_model
    cfg.openai_model = args.model
    cfg.min_person_freq = args.min_person_freq
    cfg.max_pages = args.max_pages

    if not cfg.use_local and not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY required for --cloud mode")
        print("Use --local for free local inference with Ollama")
        sys.exit(1)

    main(args.pdf)
