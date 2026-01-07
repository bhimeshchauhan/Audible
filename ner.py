#!/usr/bin/env python3
"""
Schema-free relationship inference from book PDFs (text PDFs, no OCR).

Pipeline:
Stage 0: segment into rolling scene-like windows
Stage 1: NER + aliasing + coref to canonical character IDs
Stage 2: build interaction graph (co-occurrence + evidence)
Stage 3: schema-free relationship "Describe -> Verify -> Cluster"
Stage 4: temporal smoothing of cluster assignments (EMA)  (kept as utility)
Stage 5: graphs + JSONL audit log

No training, no hardcoded relationship types. Clusters are emergent "types".
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

from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

from openai import OpenAI


# =========================
# Config
# =========================

@dataclass
class Config:
    # PDF
    max_pages: Optional[int] = None

    # spaCy
    spacy_model: str = "en_core_web_sm"
    enable_coref: bool = True
    coref_model_arch: str = "FCoref"

    # Segmentation (scene-like rolling windows)
    min_sentence_len: int = 10
    window_sentences: int = 8
    window_stride: int = 4
    max_context_chars: int = 1600

    # Characters
    min_person_freq: int = 2
    max_people_per_window: int = 10

    # Candidate selection
    min_interaction_weight: int = 2
    evidence_per_pair: int = 5
    max_pairs_to_describe: int = 350  # keep runtime bounded

    # LLM
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm_temperature: float = 0.0

    # Verification rules
    min_quote_words: int = 6
    max_quote_words: int = 35
    require_quotes: bool = True

    # Embeddings + clustering
    embed_model: str = "all-MiniLM-L6-v2"
    clustering_distance_threshold: float = 0.35  # lower => more clusters (stricter)
    min_cluster_size: int = 3

    # Outputs
    out_dir: str = "results"

    # Concurrency / throttling
    max_workers: int = int(os.getenv("OPENAI_MAX_WORKERS", "6"))
    min_request_interval_sec: float = float(os.getenv("OPENAI_MIN_REQUEST_INTERVAL_SEC", "0.25"))
    max_retries: int = int(os.getenv("OPENAI_MAX_RETRIES", "6"))


cfg = Config()


# =========================
# Utilities
# =========================

SURROGATE_RE = re.compile(r"[\uD800-\uDFFF]")
POSSESSIVE_RE = re.compile(r"(’s|'s)$", re.I)
HONORIFICS = {
    "mr","mrs","ms","miss","dr","sir","madam","lady","lord",
    "capt","captain","rev","fr","st","saint","ser"
}

def out_path(filename: str) -> str:
    return os.path.join(cfg.out_dir, filename)

def ensure_results_dir() -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

def sanitize_unicode(text: str) -> str:
    if not text:
        return ""
    text = SURROGATE_RE.sub("", text)
    text = "".join(ch for ch in text if ch in ("\n", "\t") or (0x20 <= ord(ch) <= 0x10FFFF))
    return text.encode("utf-8", "replace").decode("utf-8", "replace")

def clean_text(s: str) -> str:
    s = sanitize_unicode(s)
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def log_score(x: int) -> float:
    return 1.0 + math.log(1 + x)


# =========================
# PDF extraction (no OCR)
# =========================

def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int]) -> str:
    print("[INFO] Extracting text with PyPDF2 (no OCR)...")
    out: List[str] = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages = reader.pages if max_pages is None else reader.pages[:max_pages]
        for page in pages:
            t = page.extract_text() or ""
            if t.strip():
                out.append(t)
    return clean_text("\n".join(out))


# =========================
# spaCy + segmentation
# =========================

def build_nlp():
    nlp = spacy.load(cfg.spacy_model)
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)
    if cfg.enable_coref and "fastcoref" not in nlp.pipe_names:
        nlp.add_pipe("fastcoref", config={"model_architecture": cfg.coref_model_arch})
    return nlp

def split_sentences(nlp, text: str) -> List[str]:
    doc = nlp(text)
    sents: List[str] = []
    for s in doc.sents:
        t = s.text.strip()
        if len(t) >= cfg.min_sentence_len:
            sents.append(t)
    return sents

def windows(sentences: List[str]) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i:i+cfg.window_sentences])
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if chunk:
            out.append((i, chunk[:cfg.max_context_chars]))
        i += cfg.window_stride
    return out


# =========================
# Coref map (robust across versions)
# =========================

def coref_map(doc) -> Dict[Tuple[int, int], str]:
    mapping: Dict[Tuple[int, int], str] = {}
    clusters = getattr(doc._, "coref_clusters", None)
    if not clusters:
        return mapping

    # Newer fastcoref objects
    if hasattr(clusters[0], "mentions"):
        for cl in clusters:
            rep = cl.main.text if getattr(cl, "main", None) is not None else cl.mentions[0].text
            for m in cl.mentions:
                mapping[(m.start_char, m.end_char)] = rep
        return mapping

    # Fallback formats
    def to_span(m):
        if hasattr(m, "start_char") and hasattr(m, "end_char"):
            return m
        if isinstance(m, (list, tuple)) and len(m) >= 2 and isinstance(m[0], int) and isinstance(m[1], int):
            a, b = m[0], m[1]
            if 0 <= a < len(doc) and 0 < b <= len(doc) and a < b:
                return doc[a:b]
        if isinstance(m, dict) and "start" in m and "end" in m:
            a, b = m["start"], m["end"]
            if 0 <= a < len(doc) and 0 < b <= len(doc) and a < b:
                return doc[a:b]
        return None

    for cl in clusters:
        if not cl:
            continue
        rep_span = to_span(cl[0])
        if not rep_span:
            continue
        rep = rep_span.text
        for m in cl:
            sp = to_span(m)
            if sp:
                mapping[(sp.start_char, sp.end_char)] = rep
    return mapping


# =========================
# Character layer
# =========================

def normalize_name(name: str) -> str:
    name = sanitize_unicode(name).strip()
    name = POSSESSIVE_RE.sub("", name)
    name = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", name)
    name = re.sub(r"[^A-Za-z.\- ’']+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    parts = name.split()
    if parts and parts[0].rstrip(".").lower() in HONORIFICS:
        parts = parts[1:]

    name = " ".join(parts).strip()
    if len(name) < 2:
        return ""
    return " ".join([p[:1].upper() + p[1:] for p in name.split()])

def build_alias_map(freq: Counter[str]) -> Dict[str, str]:
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

def apply_alias(name: str, alias: Dict[str, str]) -> str:
    parts = name.split()
    if len(parts) == 1 and parts[0] in alias:
        return alias[parts[0]]
    return name

def discover_characters(nlp, sentences: List[str]) -> Tuple[Dict[str, str], Set[str]]:
    print("[INFO] Discovering characters (spaCy PERSON NER)...")
    freq: Counter[str] = Counter()
    for doc in tqdm(nlp.pipe(sentences, batch_size=64, disable=["fastcoref"]), total=len(sentences), desc="NER"):
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                nm = normalize_name(ent.text)
                if nm:
                    freq[nm] += 1

    alias = build_alias_map(freq)

    merged: Counter[str] = Counter()
    for nm, c in freq.items():
        merged[apply_alias(nm, alias)] += c

    chars: Set[str] = {n for n, c in merged.items() if c >= cfg.min_person_freq}

    print(f"[INFO] Characters (freq >= {cfg.min_person_freq}): {len(chars)}")
    for n, c in merged.most_common(20):
        print(f"  {n}: {c}{' *' if n in chars else ''}")

    return alias, chars

def chars_in_window(nlp, text: str, alias: Dict[str, str], chars: Set[str]) -> List[str]:
    doc = nlp(text)
    found: List[str] = []

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            nm = apply_alias(normalize_name(ent.text), alias)
            if nm in chars:
                found.append(nm)

    cmap = coref_map(doc)
    for rep in cmap.values():
        nm = apply_alias(normalize_name(rep), alias)
        if nm in chars:
            found.append(nm)

    uniq: List[str] = []
    seen: Set[str] = set()
    for x in found:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


# =========================
# Interaction graph
# =========================

def extract_evidence_sentences(nlp, window_text: str, a: str, b: str, k: int) -> List[str]:
    doc = nlp(window_text)
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

    out: List[str] = []
    seen: Set[str] = set()
    for e in ev:
        if e not in seen:
            seen.add(e)
            out.append(e)
        if len(out) >= k:
            break
    return out

def build_interaction_graph(nlp, win_list, alias, chars):
    edges: Counter[Tuple[str, str]] = Counter()
    evidence: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for _, wtxt in tqdm(win_list, desc="Interact"):
        present = chars_in_window(nlp, wtxt, alias, chars)
        if len(present) < 2:
            continue
        if len(present) > cfg.max_people_per_window:
            present = present[:cfg.max_people_per_window]

        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                key = (a, b) if a < b else (b, a)
                edges[key] += 1
                if len(evidence[key]) < cfg.evidence_per_pair:
                    ev = extract_evidence_sentences(nlp, wtxt, a, b, cfg.evidence_per_pair)
                    for e in ev:
                        if len(evidence[key]) >= cfg.evidence_per_pair:
                            break
                        if e not in evidence[key]:
                            evidence[key].append(e)

    G = nx.Graph()
    for (a, b), w in edges.items():
        if w < cfg.min_interaction_weight:
            continue
        G.add_edge(a, b, weight=int(w), title="\n".join(evidence.get((a, b), [])))

    return G, edges, evidence

def write_graph(G: nx.Graph, out_html: str, labeler=None):
    net = Network(height="850px", width="100%", directed=isinstance(G, nx.DiGraph))
    net.barnes_hut()

    node_w: Counter[str] = Counter()
    for u, v, d in G.edges(data=True):
        w = int(d.get("weight", 1))
        node_w[u] += w
        node_w[v] += w

    for n, w in node_w.items():
        net.add_node(n, label=n, value=8 + 3 * log_score(int(w)))

    for u, v, d in G.edges(data=True):
        w = int(d.get("weight", 1))
        title = d.get("title", "")
        lab = labeler(u, v, d) if labeler else str(w)
        net.add_edge(u, v, value=1 + log_score(w), label=lab, title=title)

    net.write_html(out_html)
    print(f"[DONE] Saved graph: {out_html}")


# =========================
# Stage 3: Describe -> Verify -> Cluster (schema-free)
# =========================

def build_prompt(a: str, b: str, evidence: List[str]) -> str:
    ev_text = "\n".join([f"- {s}" for s in evidence])
    return f"""
You are analyzing a novel. Infer the relationship between two characters ONLY from the evidence below.

Character A: {a}
Character B: {b}

Evidence (verbatim from the book):
{ev_text}

Return ONLY valid JSON with:
{{
  "a": "{a}",
  "b": "{b}",
  "description": "one short sentence describing their relationship/dynamic in plain English",
  "quotes": ["exact quote 1", "exact quote 2"]
}}

Rules:
- The description must be supported by the evidence.
- Quotes MUST be copied verbatim from the evidence lines above.
- If you cannot infer a meaningful relationship/dynamic, return:
  {{
    "a": "{a}", "b": "{b}", "description": "NONE", "quotes": []
  }}
""".strip()

def _parse_first_json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def call_openai_one(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    attempt: int = 0,
) -> Optional[dict]:
    """
    One request with exponential backoff on rate limits/transient failures.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.llm_temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        return _parse_first_json(text)
    except Exception as e:
        if attempt >= cfg.max_retries:
            print(f"[WARN] OpenAI failed after retries: {e}")
            return None
        # backoff: 1, 2, 4, 8...
        sleep_s = min(30.0, (2 ** attempt))
        time.sleep(sleep_s)
        return call_openai_one(client, model, prompt, attempt=attempt + 1)

def call_openai_concurrent(
    client: OpenAI,
    model: str,
    prompts: List[str],
) -> List[Optional[dict]]:
    """
    Concurrent “batch” calls. Preserves order of prompts in output.
    """
    results: List[Optional[dict]] = [None] * len(prompts)

    # crude pacing: spread starts out slightly to avoid spiking QPS
    def task(ix: int, p: str) -> Tuple[int, Optional[dict]]:
        time.sleep(ix * cfg.min_request_interval_sec)
        return ix, call_openai_one(client, model, p)

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futs = [ex.submit(task, i, p) for i, p in enumerate(prompts)]
        for fut in as_completed(futs):
            ix, obj = fut.result()
            results[ix] = obj

    return results

def verify_description(obj: Any, a: str, b: str, evidence: List[str]) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    if obj.get("a") != a or obj.get("b") != b:
        return None

    desc = (obj.get("description") or "").strip()
    quotes = obj.get("quotes", [])

    if not desc or desc.upper() == "NONE":
        return None

    if cfg.require_quotes:
        if not isinstance(quotes, list) or len(quotes) == 0:
            return None

        ev_join = "\n".join(evidence)
        good_quotes: List[str] = []
        for q in quotes:
            if not isinstance(q, str):
                continue
            q2 = q.strip()
            if not q2:
                continue
            wc = len(q2.split())
            if wc < cfg.min_quote_words or wc > cfg.max_quote_words:
                continue
            if q2 in ev_join:
                good_quotes.append(q2)

        if not good_quotes:
            return None

        quotes = good_quotes[:2]

    return {
        "a": a,
        "b": b,
        "description": desc,
        "quotes": quotes,
        "evidence": evidence,
    }

def cluster_descriptions(descriptions: List[str], embedder: SentenceTransformer):
    X = embedder.encode(descriptions, normalize_embeddings=True, show_progress_bar=True)
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=cfg.clustering_distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clusterer.fit_predict(X)
    return labels, X

def summarize_clusters(events: List[dict], labels: List[int]) -> dict:
    by: Dict[int, List[dict]] = defaultdict(list)
    for ev, lab in zip(events, labels):
        by[int(lab)].append(ev)

    summary: dict = {}
    for lab, items in by.items():
        if len(items) < cfg.min_cluster_size:
            continue
        ex = items[: min(6, len(items))]
        summary[str(lab)] = {
            "count": len(items),
            "examples": [
                {
                    "a": e["a"], "b": e["b"],
                    "description": e["description"],
                    "quotes": e["quotes"],
                } for e in ex
            ]
        }
    return summary


def main(pdf_path: str):
    ensure_results_dir()

    out_events_jsonl = out_path("relationship_events.jsonl")
    out_cluster_summary = out_path("cluster_summary.json")
    out_interaction_html = out_path("interaction_graph.html")
    out_cluster_graph_html = out_path("relationship_clusters_graph.html")

    nlp = build_nlp()
    client = OpenAI()

    text = extract_text_from_pdf(pdf_path, cfg.max_pages)
    sentences = split_sentences(nlp, text)
    print(f"[INFO] Sentences kept: {len(sentences)}")

    alias, chars = discover_characters(nlp, sentences)
    win_list = windows(sentences)
    print(f"[INFO] Windows: {len(win_list)}")

    Gi, edge_counts, edge_evidence = build_interaction_graph(nlp, win_list, alias, chars)
    if Gi.number_of_edges() == 0:
        print("[ERROR] No interaction edges found. Lower min_person_freq or min_interaction_weight.")
        sys.exit(2)

    write_graph(Gi, out_interaction_html)

    candidates = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)
    candidates = candidates[:cfg.max_pairs_to_describe]

    # Build prompts
    prompts: List[str] = []
    meta: List[Tuple[str, str, int, List[str]]] = []

    print("[INFO] Building prompts...")
    for (a, b), w in candidates:
        ev = edge_evidence.get((a, b), []) or edge_evidence.get((b, a), [])
        if not ev:
            continue
        prompts.append(build_prompt(a, b, ev))
        meta.append((a, b, int(w), ev))

    print(f"[INFO] Calling OpenAI concurrently: {len(prompts)} prompts, max_workers={cfg.max_workers}")
    responses = call_openai_concurrent(client, cfg.openai_model, prompts)

    # Verify
    events: List[dict] = []
    for resp, (a, b, w, ev) in zip(responses, meta):
        verified = verify_description(resp, a, b, ev)
        if not verified:
            continue
        verified["pair"] = f"{a}|||{b}"
        verified["interaction_weight"] = int(w)
        events.append(verified)

    if not events:
        print("[ERROR] No verified relationship descriptions. Increase window size or loosen quote constraints.")
        sys.exit(3)

    print("[INFO] Embedding + clustering descriptions...")
    embedder = SentenceTransformer(cfg.embed_model)
    descriptions = [e["description"] for e in events]
    cluster_labels, _ = cluster_descriptions(descriptions, embedder)

    # Save events
    with open(out_events_jsonl, "w", encoding="utf-8") as f:
        for e, lab in zip(events, cluster_labels):
            rec = dict(e)
            rec["cluster"] = int(lab)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[DONE] Saved events: {out_events_jsonl}")

    # Cluster summary
    summary = summarize_clusters(events, cluster_labels)
    with open(out_cluster_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved cluster summary: {out_cluster_summary}")

    # Cluster graph
    Gc = nx.Graph()
    ec: Counter[Tuple[str, str, int]] = Counter()
    for e, lab in zip(events, cluster_labels):
        a, b = e["a"], e["b"]
        key = (a, b, int(lab)) if a < b else (b, a, int(lab))
        ec[key] += 1

    for (a, b, lab), w in ec.items():
        Gc.add_edge(a, b, weight=int(w), cluster=str(lab), title="")

    def lab(u, v, d):
        return f"cluster {d.get('cluster')} ({d.get('weight')})"

    write_graph(Gc, out_cluster_graph_html, labeler=lab)

    print("[INFO] Done. You now have:")
    print(f"  - {out_interaction_html}")
    print(f"  - {out_cluster_graph_html}")
    print(f"  - {out_events_jsonl}")
    print(f"  - {out_cluster_summary}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--model", default=cfg.openai_model)
    ap.add_argument("--window_sentences", type=int, default=cfg.window_sentences)
    ap.add_argument("--window_stride", type=int, default=cfg.window_stride)
    ap.add_argument("--min_person_freq", type=int, default=cfg.min_person_freq)
    ap.add_argument("--min_interaction_weight", type=int, default=cfg.min_interaction_weight)
    ap.add_argument("--cluster_threshold", type=float, default=cfg.clustering_distance_threshold)
    args = ap.parse_args()

    cfg.openai_model = args.model
    cfg.window_sentences = args.window_sentences
    cfg.window_stride = args.window_stride
    cfg.min_person_freq = args.min_person_freq
    cfg.min_interaction_weight = args.min_interaction_weight
    cfg.clustering_distance_threshold = args.cluster_threshold

    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set. Put it in .env (uncommitted) or environment.")
        sys.exit(1)

    main(args.pdf)
