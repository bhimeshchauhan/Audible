# 📚➡️🎧 PDF → Cinematic Audiobook  
### Character-Aware Voices · Emotion · Natural Flow

Transform a **book PDF** into a **realistic, listenable audiobook** — with consistent character voices, sentiment-aware delivery, and natural pacing.

This repository is built around one guiding principle:

> **Audiobooks are not text-to-speech. They are performance.**

---

## ✨ Why This Exists

Most automated audiobook attempts fail because:

- Voices change mid-book
- Dialogue and narration sound identical
- No rhythm, emotion, or pacing
- Characters feel flat and interchangeable

This project solves that by treating audiobook generation as a **structured NLP + audio pipeline**, not a single TTS call.

---

## 🧠 Core Idea

We first **understand the book**, then **perform it**.

That means:
- Understanding **who is speaking**
- Understanding **how they feel**
- Preserving **voice consistency**
- Shaping **delivery, pauses, and emphasis**
- Generating audio **incrementally and safely**

---

## 🏗️ High-Level Architecture

### Stage A — PDF Ingestion & Normalization
**Input:** text-based PDF (no OCR required)

- Extract text via PyPDF2
- Remove headers, footers, line breaks, hyphenation
- Normalize quotes and punctuation
- Optional chapter detection via heuristics

**Output:** normalized text

---

### Stage B — Scene-Level Segmentation
Audiobooks work best on **coherent windows**, not whole chapters.

- Rolling “scene-like” windows  
- Example: 6–10 sentences, stride 3–5  
- Preserves narrative flow and temporal order

**Output:** `scenes.jsonl`

---

### Stage C — Character Discovery & Coreference
Goal: **stable character identities** across the entire book.

- Named Entity Recognition (`PERSON`)
- Alias merging (e.g. *Elizabeth* ↔ *Liz*)
- Coreference resolution (*he*, *she*, *the doctor*)
- Frequency thresholds to filter noise

**Outputs:**
- `characters.json`
- `mentions.jsonl`

---

### Stage D — Dialogue & Speaker Attribution
We distinguish:

- **Narration**
- **Dialogue**
- **Internal monologue** (optional)

Speaker inference uses:
- Quote detection
- Attribution verbs (“said”, “asked”, “replied”)
- Coreference-aware nearest-speaker logic
- Safe fallback to narrator if uncertain

**Output:** `utterances.jsonl`

---

### Stage E — Emotion & Delivery Modeling
This is where realism emerges.

For each utterance we infer:
- Sentiment / polarity
- Emotional intensity
- Speaking intent (calm, tense, playful, angry)
- Pacing and pause structure

These become **delivery directives** for TTS:
- Pace
- Energy
- Emphasis points
- Pause length

**Output:** `delivery.jsonl`

---

### Stage F — Voice Casting (Consistency First)
Each character is assigned **one stable voice**.

- Narrator voice (global)
- One voice per character
- Optional hints: age, tone, temperament
- No mid-book voice drift

**Output:** `voice_map.json`

---

### Stage G — Chunked Audio Synthesis
Audiobooks are long. We generate audio safely.

- One utterance → one audio job
- Content-hash caching
- Rate-limit-safe batching
- Retry-friendly execution

**Outputs:**
- `results/audio_chunks/`
- `results/audio_manifest.json`

---

### Stage H — Stitching & Mastering
Final polish:

- Order-preserving merge
- Loudness normalization
- Micro-fades between chunks
- Optional chapter-level exports

**Output:**
- `results/audiobook.mp3`
- `results/chapters/*.mp3`

---

## 🔁 Data Flow Overview

```text
PDF
 └─> Text Extraction
     └─> Scene Segmentation
         └─> Character + Coref
             └─> Dialogue Attribution
                 └─> Emotion Modeling
                     └─> Voice Casting
                         └─> Chunked TTS
                             └─> Stitching
                                 └─> Audiobook
```

## 🎭 What Makes It Feel Real

✅ Voice Consistency
Characters never change voices mid-story.

✅ Context-Aware Delivery
Emotion and intent shape how lines are spoken.

✅ Natural Pacing
Pauses, emphasis, and transitions feel human.

✅ Narrative Structure Preserved
Scenes and chapters flow naturally.

## 🧪 Analysis Layer (Character Graphs)

- The repo currently includes a powerful analysis engine (ner.py) that:
- Builds interaction graphs
- Infers relationship dynamics
- Clusters emergent relationship types
- Produces auditable evidence with quotes

### Outputs:

- interaction_graph.html
- relationship_clusters_graph.html
- relationship_events.jsonl
- cluster_summary.json

## 🚀 Quickstart

### Environment Setup
```python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Create .env (not committed):
```shell
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

### Run Analysis
```shell
python ner.py path/to/book.pdf
```

Artifacts are written to results/ (git-ignored).

```text
📁 Recommended Repo Structure
.
├─ ner.py                      # Character + relationship inference
├─ pipeline/                   # (future) full audiobook pipeline
│  ├─ extract.py
│  ├─ segment.py
│  ├─ characters.py
│  ├─ dialogue.py
│  ├─ emotion.py
│  ├─ voices.py
│  ├─ tts.py
│  └─ stitch.py
├─ results/                    # Generated outputs (ignored)
├─ requirements.txt
├─ .env.example
└─ README.md
```

## 🧭 Design Principles

- Auditability: every stage writes inspectable artifacts

- Determinism: stable voices, caching, repeatable runs

- Fail-safe: narrator fallback when uncertain

- Schema-free: relationships emerge, not hardcoded

- Bounded runtime: batching + candidate limits

## 🗺️ Roadmap

 - Robust chapter & TOC parsing

 - Improved speaker attribution models

 - Emotion smoothing across scenes

 - Provider-specific prosody tuning

 - Automatic voice casting via embeddings

 - One-command CLI: pdf2audiobook book.pdf

## ⚠️ Notes & Constraints

Text-extractable PDFs only (OCR not yet included)

Speaker attribution is probabilistic

Audio realism depends on TTS provider controls

## 📜 License

[GNU General Public License](LICENSE)