#!/usr/bin/env python3
"""
High-precision relationship extraction from text PDFs (no OCR).
Uses:
- PyPDF2 for text
- spaCy for sentence splitting + PERSON NER
- fastcoref for coreference resolution (optional but helps)
- Regex patterns for explicit relationship statements (high accuracy)

Outputs:
- extracted_relations.jsonl
- character_graph.html

Install:
  pip install PyPDF2 spacy tqdm pyvis networkx fastcoref
  python -m spacy download en_core_web_sm

Run:
  python ner.py novel.pdf
"""

import os, re, sys, json, math
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

import PyPDF2
import spacy
from tqdm import tqdm
from pyvis.network import Network

from fastcoref import spacy_component  # registers "fastcoref"


@dataclass
class Config:
    max_pages: Optional[int] = None
    min_sentence_len: int = 10

    spacy_model: str = "en_core_web_sm"
    enable_coref: bool = True
    coref_model_arch: str = "FCoref"

    min_person_freq: int = 2
    max_persons_per_sentence: int = 12

    evidence_window_sentences: int = 3  # helps catch "X ... His father Y ..."
    output_jsonl: str = "extracted_relations.jsonl"
    output_html: str = "character_graph.html"

CFG = Config()


# ---------- Unicode sanitize ----------
def sanitize_unicode(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\uD800-\uDFFF]", "", text)
    text = "".join(ch for ch in text if ch in ("\n", "\t") or (0x20 <= ord(ch) <= 0x10FFFF))
    return text.encode("utf-8", "replace").decode("utf-8", "replace")

def clean_text(s: str) -> str:
    s = sanitize_unicode(s)
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ---------- PDF extraction ----------
def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> str:
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


# ---------- spaCy init ----------
def build_nlp():
    nlp = spacy.load(CFG.spacy_model)
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)
    if CFG.enable_coref and "fastcoref" not in nlp.pipe_names:
        nlp.add_pipe("fastcoref", config={"model_architecture": CFG.coref_model_arch})
    return nlp


# ---------- sentence splitting ----------
def get_sentences(nlp, text: str) -> List[str]:
    doc = nlp(text)
    sents = []
    for s in doc.sents:
        t = s.text.strip()
        if len(t) >= CFG.min_sentence_len:
            sents.append(t)
    return sents

def collect_window(sentences: List[str], i: int, k: int) -> str:
    chunk = " ".join(sentences[i:i+k])
    chunk = re.sub(r"\s+", " ", chunk).strip()
    return chunk


# ---------- name normalization + aliasing ----------
HONORIFICS = {"mr","mrs","ms","miss","dr","sir","madam","lady","lord","capt","captain","rev","fr","st","saint","ser"}
POSSESSIVE_RE = re.compile(r"(’s|'s)$", re.I)

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
    name = " ".join([p[:1].upper() + p[1:] for p in name.split()])
    return name

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
        # dominant full name for this last name
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


# ---------- coref helper ----------
def coref_map(doc) -> Dict[Tuple[int,int], str]:
    mapping = {}
    clusters = getattr(doc._, "coref_clusters", None)
    if not clusters:
        return mapping

    # Support both object and list formats
    if hasattr(clusters[0], "mentions"):
        for cl in clusters:
            rep = cl.main.text if getattr(cl, "main", None) is not None else cl.mentions[0].text
            for m in cl.mentions:
                mapping[(m.start_char, m.end_char)] = rep
        return mapping

    # list-format clusters
    def to_span(m):
        if hasattr(m, "start_char"):
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


# ---------- character extraction ----------
def extract_characters(nlp, sentences: List[str]):
    persons_per_sentence = []
    freq = Counter()

    print("[INFO] Running spaCy NER...")
    for doc in tqdm(nlp.pipe(sentences, batch_size=64, disable=["fastcoref"]), total=len(sentences), desc="spaCy"):
        ppl = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                nm = normalize_name(ent.text)
                if nm:
                    ppl.append(nm)
        uniq = []
        seen = set()
        for p in ppl:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        if len(uniq) > CFG.max_persons_per_sentence:
            uniq = []
        for p in uniq:
            freq[p] += 1
        persons_per_sentence.append(uniq)

    alias = build_alias_map(freq)

    merged_pps = []
    merged_freq = Counter()
    for lst in persons_per_sentence:
        merged = []
        seen = set()
        for p in lst:
            p2 = apply_alias(p, alias)
            if p2 and p2 not in seen:
                seen.add(p2)
                merged.append(p2)
                merged_freq[p2] += 1
        merged_pps.append(merged)

    # Strong merge: if "Button" exists, map it to dominant "* Button"
    if "Button" in merged_freq:
        full_buttons = [(n, merged_freq[n]) for n in merged_freq if n.endswith(" Button") and len(n.split()) >= 2]
        full_buttons.sort(key=lambda x: x[1], reverse=True)
        if full_buttons:
            top, topc = full_buttons[0]
            secondc = full_buttons[1][1] if len(full_buttons) > 1 else 0
            if topc >= 2 * max(1, secondc):
                alias["Button"] = top

    chars = {n for n, c in merged_freq.items() if c >= CFG.min_person_freq}

    print(f"[INFO] Characters (freq >= {CFG.min_person_freq}): {len(chars)}")
    for n, c in merged_freq.most_common(20):
        print(f"  {n}: {c}{' *' if n in chars else ''}")

    return merged_pps, merged_freq, alias, chars


# ---------- explicit relationship patterns (high precision) ----------
# We only extract when the text matches these literal templates.
PATTERNS = [
    # X, son of Y / X was the son of Y
    ("son_of", re.compile(r"\b(?P<A>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b.*?\bson of\b.*?\b(?P<B>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")),
    # X, daughter of Y
    ("daughter_of", re.compile(r"\b(?P<A>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b.*?\bdaughter of\b.*?\b(?P<B>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")),
    # X's father Y / X's mother Y  (often "Benjamin Button's father, Roger Button")
    ("father", re.compile(r"\b(?P<A>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)['’]s\s+father\b.*?\b(?P<B>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")),
    ("mother", re.compile(r"\b(?P<A>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)['’]s\s+mother\b.*?\b(?P<B>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")),
    # Y, the father of X / mother of
    ("father_of", re.compile(r"\b(?P<B>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b.*?\bthe father of\b.*?\b(?P<A>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")),
    ("mother_of", re.compile(r"\b(?P<B>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b.*?\bthe mother of\b.*?\b(?P<A>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")),
    # married
    ("married", re.compile(r"\b(?P<A>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b.*?\bmarried\b.*?\b(?P<B>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")),
    # wife/husband
    ("wife", re.compile(r"\b(?P<A>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)['’]s\s+wife\b.*?\b(?P<B>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")),
    ("husband", re.compile(r"\b(?P<A>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)['’]s\s+husband\b.*?\b(?P<B>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")),
]

def canon(name: str, alias: Dict[str,str]) -> str:
    n = normalize_name(name)
    if not n:
        return ""
    return apply_alias(n, alias)

def extract_relations_from_evidence(evidence: str, alias: Dict[str,str], characters: set) -> List[dict]:
    rels = []
    for reltype, rx in PATTERNS:
        for m in rx.finditer(evidence):
            A = canon(m.group("A"), alias)
            B = canon(m.group("B"), alias)
            if not A or not B:
                continue
            if A not in characters or B not in characters:
                continue

            # normalize direction into (subject -> object) with a consistent relation label
            if reltype in ("father", "mother"):
                # "A's father B" means B is father of A
                label = "father_of" if reltype == "father" else "mother_of"
                subj, obj = B, A
            elif reltype in ("son_of", "daughter_of"):
                # "A son of B" means B parent_of A (but keep label as written)
                label = reltype
                subj, obj = A, B
            else:
                label = reltype
                subj, obj = A, B

            rels.append({
                "a": subj,
                "b": obj,
                "relation": label,
                "evidence": evidence
            })
    return rels


# ---------- graph ----------
def score(x: int) -> float:
    return 1.0 + math.log(1 + x)

def build_graph(edges: Counter, out_html: str):
    net = Network(directed=True, height="800px", width="100%")
    net.barnes_hut()

    node_w = Counter()
    for (a, rel, b), c in edges.items():
        node_w[a] += c
        node_w[b] += c

    for n, c in node_w.items():
        net.add_node(n, label=n, value=8 + 3 * score(c))

    for (a, rel, b), c in edges.items():
        net.add_edge(a, b, label=f"{rel} ({c})", title=f"{a} → {b}\n{rel}\ncount={c}", value=1 + score(c))

    net.write_html(out_html)
    print(f"[DONE] Saved graph: {out_html}")


def main(pdf_path: str):
    if not os.path.exists(pdf_path):
        print("[ERROR] File not found")
        sys.exit(1)

    nlp = build_nlp()

    text = extract_text_from_pdf(pdf_path, CFG.max_pages)
    if not text:
        print("[ERROR] No text extracted.")
        sys.exit(2)

    sentences = get_sentences(nlp, text)
    print(f"[INFO] Sentences kept: {len(sentences)}")

    _, _, alias, characters = extract_characters(nlp, sentences)

    # relation extraction
    extracted = []
    edge_counts = Counter()

    print("[INFO] Extracting explicit relationships using patterns...")
    for i in tqdm(range(len(sentences)), desc="Scan"):
        evidence = collect_window(sentences, i, CFG.evidence_window_sentences)
        if not evidence:
            continue
        rels = extract_relations_from_evidence(evidence, alias, characters)
        for r in rels:
            extracted.append(r)
            edge_counts[(r["a"], r["relation"], r["b"])] += 1

    if not extracted:
        print("[ERROR] No explicit relationships found with current patterns.")
        print("Next step: expand pattern set (still grounded) for the book’s phrasing.")
        sys.exit(3)

    with open(CFG.output_jsonl, "w", encoding="utf-8") as f:
        for r in extracted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE] Saved: {CFG.output_jsonl}")

    print("[INFO] Top edges:")
    for (a, rel, b), c in edge_counts.most_common(20):
        print(f"  {a} --{rel}--> {b} (x{c})")

    build_graph(edge_counts, CFG.output_html)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ner.py novel.pdf")
        sys.exit(1)
    main(sys.argv[1])
