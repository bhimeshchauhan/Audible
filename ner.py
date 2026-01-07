#!/usr/bin/env python3
"""
Schema-free relationship inference from book PDFs (text PDFs, no OCR).

Pipeline:
Stage 0: segment into rolling scene-like windows
Stage 1: NER + aliasing + coref to canonical character IDs
Stage 2: build interaction graph (co-occurrence + evidence)
Stage 3: schema-free relationship "Describe -> Verify -> Cluster"
Stage 4: temporal smoothing of cluster assignments (EMA)
Stage 5: graphs + JSONL audit log

No training, no hardcoded relationship types. Clusters are emergent "types".

Requires OpenAI for reliable relationship descriptions. (No training.)
"""

import os, re, sys, json, math, argparse
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

import PyPDF2
import spacy
import networkx as nx
from tqdm import tqdm
from pyvis.network import Network
from fastcoref import spacy_component  # registers "fastcoref"

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from openai import OpenAI
from dotenv import load_dotenv
import time
load_dotenv()

# =========================
# Config
# =========================

@dataclass
class CFG:
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
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # Verification rules
    min_quote_words: int = 6
    max_quote_words: int = 35
    require_quotes: bool = True
    max_descriptions_per_pair: int = 2  # self-consistency: 2 different windows

    # Embeddings + clustering
    embed_model: str = "all-MiniLM-L6-v2"
    clustering_distance_threshold: float = 0.35  # lower => more clusters (stricter)
    min_cluster_size: int = 3

    # Temporal smoothing
    ema_alpha: float = 0.35

    # Outputs
    out_dir: str = "results"
    out_events_jsonl: str = "results/relationship_events.jsonl"
    out_cluster_summary: str = "results/cluster_summary.json"
    out_interaction_html: str = "results/interaction_graph.html"
    out_cluster_graph_html: str = "results/relationship_clusters_graph.html"

CFG = CFG()


# =========================
# Utilities
# =========================

SURROGATE_RE = re.compile(r"[\uD800-\uDFFF]")
POSSESSIVE_RE = re.compile(r"(’s|'s)$", re.I)
HONORIFICS = {"mr","mrs","ms","miss","dr","sir","madam","lady","lord","capt","captain","rev","fr","st","saint","ser"}

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

def ensure_results_dir():
    os.makedirs(CFG.out_dir, exist_ok=True)


# =========================
# PDF extraction (no OCR)
# =========================

def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int]) -> str:
    print("[INFO] Extracting text with PyPDF2 (no OCR)...")
    out = []
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
    nlp = spacy.load(CFG.spacy_model)
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)
    if CFG.enable_coref and "fastcoref" not in nlp.pipe_names:
        nlp.add_pipe("fastcoref", config={"model_architecture": CFG.coref_model_arch})
    return nlp

def split_sentences(nlp, text: str) -> List[str]:
    doc = nlp(text)
    sents = []
    for s in doc.sents:
        t = s.text.strip()
        if len(t) >= CFG.min_sentence_len:
            sents.append(t)
    return sents

def windows(sentences: List[str]) -> List[Tuple[int, str]]:
    out = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i:i+CFG.window_sentences])
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if chunk:
            out.append((i, chunk[:CFG.max_context_chars]))
        i += CFG.window_stride
    return out


# =========================
# Coref map (robust across versions)
# =========================

def coref_map(doc) -> Dict[Tuple[int,int], str]:
    mapping = {}
    clusters = getattr(doc._, "coref_clusters", None)
    if not clusters:
        return mapping

    if hasattr(clusters[0], "mentions"):
        for cl in clusters:
            rep = cl.main.text if getattr(cl, "main", None) is not None else cl.mentions[0].text
            for m in cl.mentions:
                mapping[(m.start_char, m.end_char)] = rep
        return mapping

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

def build_alias_map(freq: Counter) -> Dict[str, str]:
    full = [n for n in freq if len(n.split()) >= 2]
    by_last = defaultdict(list)
    for fn in full:
        by_last[fn.split()[-1]].append(fn)

    alias = {}
    for last, cands in by_last.items():
        cands = sorted(cands, key=lambda x: freq[x], reverse=True)
        if not cands:
            continue
        top = cands[0]
        second = cands[1] if len(cands) > 1 else None
        if second is None or freq[top] >= 2 * max(1, freq[second]):
            alias[last] = top
    return alias

def apply_alias(name: str, alias: Dict[str,str]) -> str:
    parts = name.split()
    if len(parts) == 1 and parts[0] in alias:
        return alias[parts[0]]
    return name

def discover_characters(nlp, sentences: List[str]) -> Tuple[Dict[str,str], set]:
    print("[INFO] Discovering characters (spaCy PERSON NER)...")
    freq = Counter()
    for doc in tqdm(nlp.pipe(sentences, batch_size=64, disable=["fastcoref"]), total=len(sentences), desc="NER"):
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                nm = normalize_name(ent.text)
                if nm:
                    freq[nm] += 1

    alias = build_alias_map(freq)

    merged = Counter()
    for nm, c in freq.items():
        merged[apply_alias(nm, alias)] += c

    chars = {n for n, c in merged.items() if c >= CFG.min_person_freq}

    print(f"[INFO] Characters (freq >= {CFG.min_person_freq}): {len(chars)}")
    for n, c in merged.most_common(20):
        print(f"  {n}: {c}{' *' if n in chars else ''}")

    return alias, chars

def chars_in_window(nlp, text: str, alias: Dict[str,str], chars: set) -> List[str]:
    doc = nlp(text)
    found = []

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            nm = apply_alias(normalize_name(ent.text), alias)
            if nm in chars:
                found.append(nm)

    cmap = coref_map(doc)
    for (_, _), rep in cmap.items():
        nm = apply_alias(normalize_name(rep), alias)
        if nm in chars:
            found.append(nm)

    uniq, seen = [], set()
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
    ev = []
    for s in sents:
        if a in s and b in s:
            ev.append(s)
    # if none, keep some mention sentences (helps LLM)
    if not ev:
        for s in sents:
            if a in s or b in s:
                ev.append(s)
            if len(ev) >= k:
                break

    out, seen = [], set()
    for e in ev:
        if e not in seen:
            seen.add(e)
            out.append(e)
        if len(out) >= k:
            break
    return out

def build_interaction_graph(nlp, win_list, alias, chars):
    edges = Counter()
    evidence = defaultdict(list)

    for t_idx, wtxt in tqdm(win_list, desc="Interact"):
        present = chars_in_window(nlp, wtxt, alias, chars)
        if len(present) < 2:
            continue
        if len(present) > CFG.max_people_per_window:
            present = present[:CFG.max_people_per_window]

        for i in range(len(present)):
            for j in range(i+1, len(present)):
                a, b = present[i], present[j]
                key = (a, b) if a < b else (b, a)
                edges[key] += 1
                if len(evidence[key]) < CFG.evidence_per_pair:
                    ev = extract_evidence_sentences(nlp, wtxt, a, b, CFG.evidence_per_pair)
                    for e in ev:
                        if len(evidence[key]) >= CFG.evidence_per_pair:
                            break
                        if e not in evidence[key]:
                            evidence[key].append(e)

    G = nx.Graph()
    for (a, b), w in edges.items():
        if w < CFG.min_interaction_weight:
            continue
        G.add_edge(a, b, weight=int(w), title="\n".join(evidence.get((a,b), [])))

    return G, edges, evidence


def write_graph(G: nx.Graph, out_html: str, labeler=None):
    net = Network(height="850px", width="100%", directed=isinstance(G, nx.DiGraph))
    net.barnes_hut()

    node_w = Counter()
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        node_w[u] += w
        node_w[v] += w

    for n, w in node_w.items():
        net.add_node(n, label=n, value=8 + 3 * log_score(int(w)))

    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        title = d.get("title", "")
        lab = labeler(u, v, d) if labeler else str(w)
        net.add_edge(u, v, value=1 + log_score(int(w)), label=lab, title=title)

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

def call_openai_batch(
    client: OpenAI,
    model: str,
    prompts: List[str],
    sleep_s: float = 1.2,
) -> List[Optional[dict]]:
    """
    Sends prompts sequentially but in controlled batches.
    Returns parsed JSON objects or None per prompt.
    """
    results = []

    for prompt in prompts:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=CFG.llm_temperature,
            )
            text = resp.choices[0].message.content.strip()
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                results.append(None)
            else:
                results.append(json.loads(m.group(0)))
        except Exception as e:
            print(f"[WARN] OpenAI error: {e}")
            results.append(None)

        time.sleep(sleep_s)  # <-- rate limit safety

    return results

def verify_description(obj: dict, a: str, b: str, evidence: List[str]) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    if obj.get("a") != a or obj.get("b") != b:
        return None
    desc = (obj.get("description") or "").strip()
    quotes = obj.get("quotes", [])
    if not desc:
        return None
    if desc.upper() == "NONE":
        return None
    if CFG.require_quotes:
        if not isinstance(quotes, list) or len(quotes) == 0:
            return None

        # verify each quote is a verbatim substring of some evidence line
        ev_join = "\n".join(evidence)
        good_quotes = []
        for q in quotes:
            if not isinstance(q, str):
                continue
            q2 = q.strip()
            if not q2:
                continue
            wc = len(q2.split())
            if wc < CFG.min_quote_words or wc > CFG.max_quote_words:
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

    # Agglomerative clustering with distance threshold (no need to pick K)
    # distance = 1 - cosine_sim
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=CFG.clustering_distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clusterer.fit_predict(X)
    return labels, X


def summarize_clusters(events: List[dict], labels: List[int]):
    by = defaultdict(list)
    for ev, lab in zip(events, labels):
        by[int(lab)].append(ev)

    summary = {}
    for lab, items in by.items():
        # drop tiny clusters
        if len(items) < CFG.min_cluster_size:
            continue
        # show top examples
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


# =========================
# Stage 4: Temporal smoothing (EMA)
# =========================

def smooth_cluster_over_time(pair_events: List[Tuple[int, int]]) -> List[int]:
    """
    pair_events: list of (time_index, cluster_label) sorted by time.
    returns smoothed label sequence (simple EMA-like vote).
    """
    # For simplicity: keep a running score per cluster and decay.
    scores = defaultdict(float)
    out = []
    last_t = None
    for t, lab in pair_events:
        if last_t is None:
            last_t = t
        # decay for gaps
        gap = max(0, t - last_t)
        decay = (1.0 - CFG.ema_alpha) ** gap
        for k in list(scores.keys()):
            scores[k] *= decay
            if scores[k] < 1e-6:
                del scores[k]
        # update current
        scores[lab] += CFG.ema_alpha
        # choose max
        best = max(scores.items(), key=lambda x: x[1])[0]
        out.append(best)
        last_t = t
    return out


# =========================
# Build cluster graph
# =========================

def build_cluster_graph(events: List[dict], cluster_labels: List[int]) -> nx.DiGraph:
    """
    Nodes: characters
    Edges: weighted by frequency; label shows cluster id
    """
    G = nx.DiGraph()
    edge_counter = Counter()

    for ev, lab in zip(events, cluster_labels):
        a, b = ev["a"], ev["b"]
        edge_counter[(a, b, int(lab))] += 1

    for (a, b, lab), w in edge_counter.items():
        G.add_edge(
            a, b,
            weight=int(w),
            cluster=str(lab),
            title="\n".join(ev for ev in [])  # keep simple; evidence is in JSONL
        )
    return G


def main(pdf_path: str):
    ensure_results_dir()
    nlp = build_nlp()
    client = OpenAI()

    text = extract_text_from_pdf(pdf_path, CFG.max_pages)
    sentences = split_sentences(nlp, text)
    print(f"[INFO] Sentences kept: {len(sentences)}")

    alias, chars = discover_characters(nlp, sentences)
    win_list = windows(sentences)
    print(f"[INFO] Windows: {len(win_list)}")

    Gi, edge_counts, edge_evidence = build_interaction_graph(nlp, win_list, alias, chars)
    if Gi.number_of_edges() == 0:
        print("[ERROR] No interaction edges found. Lower min_person_freq or min_interaction_weight.")
        sys.exit(2)

    write_graph(Gi, CFG.out_interaction_html)

    # Candidate selection: strongest interactions first
    candidates = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)

    # Bound runtime
    candidates = candidates[:CFG.max_pairs_to_describe]

    # Stage 3: describe + verify
    print("[INFO] Describing relationships (batched, schema-free)...")

    batch_size = 8
    all_events = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]

        prompts = []
        meta = []

        for (a, b), w in batch:
            ev = edge_evidence.get((a, b), []) or edge_evidence.get((b, a), [])
            if not ev:
                continue
            prompts.append(build_prompt(a, b, ev))
            meta.append((a, b, w, ev))

        if not prompts:
            continue

        responses = call_openai_batch(client, CFG.openai_model, prompts)

        for resp, (a, b, w, ev) in zip(responses, meta):
            verified = verify_description(resp, a, b, ev)
            if not verified:
                continue

            verified["pair"] = f"{a}|||{b}"
            verified["interaction_weight"] = int(w)
            all_events.append(verified)

    events = all_events

    # Embed + cluster
    print("[INFO] Embedding + clustering descriptions (emergent relationship types)...")
    embedder = SentenceTransformer(CFG.embed_model)
    descriptions = [e["description"] for e in events]
    cluster_labels, X = cluster_descriptions(descriptions, embedder)

    # Save events
    with open(CFG.out_events_jsonl, "w", encoding="utf-8") as f:
        for e, lab in zip(events, cluster_labels):
            rec = dict(e)
            rec["cluster"] = int(lab)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[DONE] Saved events: {CFG.out_events_jsonl}")

    # Cluster summary
    summary = summarize_clusters(events, cluster_labels)
    with open(CFG.out_cluster_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved cluster summary: {CFG.out_cluster_summary}")

    # Graph: label edges by cluster id (emergent type)
    Gc = nx.Graph()
    ec = Counter()
    for e, lab in zip(events, cluster_labels):
        a, b = e["a"], e["b"]
        key = (a, b, int(lab))
        ec[key] += 1

    for (a, b, lab), w in ec.items():
        # Use undirected for "dynamic"; direction isn't reliable in schema-free
        Gc.add_edge(a, b, weight=int(w), cluster=str(lab), title="; ".join(events[:0]))

    def lab(u, v, d):
        return f"cluster {d.get('cluster')} ({d.get('weight')})"

    write_graph(Gc, CFG.out_cluster_graph_html, labeler=lab)

    print("[INFO] Done. You now have:")
    print(f"  - {CFG.out_interaction_html} (who interacts)")
    print(f"  - {CFG.out_cluster_graph_html} (emergent relationship clusters)")
    print(f"  - {CFG.out_events_jsonl} (audit log with quotes)")
    print(f"  - {CFG.out_cluster_summary} (examples per cluster)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--model", default=CFG.openai_model)
    ap.add_argument("--window_sentences", type=int, default=CFG.window_sentences)
    ap.add_argument("--window_stride", type=int, default=CFG.window_stride)
    ap.add_argument("--min_person_freq", type=int, default=CFG.min_person_freq)
    ap.add_argument("--min_interaction_weight", type=int, default=CFG.min_interaction_weight)
    ap.add_argument("--cluster_threshold", type=float, default=CFG.clustering_distance_threshold)
    args = ap.parse_args()

    CFG.openai_model = args.model
    CFG.window_sentences = args.window_sentences
    CFG.window_stride = args.window_stride
    CFG.min_person_freq = args.min_person_freq
    CFG.min_interaction_weight = args.min_interaction_weight
    CFG.clustering_distance_threshold = args.cluster_threshold

    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set.")
        sys.exit(1)

    main(args.pdf)
