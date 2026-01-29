#!/usr/bin/env python3
"""
Command-line interface for PDF character relationship extraction.

Usage:
    python cli.py path/to/book.pdf [options]

Examples:
    # Local mode (default, FREE - uses Ollama)
    python cli.py books/novel.pdf
    python cli.py books/novel.pdf --local_model mistral

    # Cloud mode (requires OPENAI_API_KEY)
    python cli.py books/novel.pdf --cloud

Setup for local:
    1. Install Ollama: brew install ollama
    2. Start: brew services start ollama
    3. Pull model: ollama pull llama3.2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import config
from backend.services.pdf_processor import (
    _build_nlp,
    _extract_text_from_pdf,
    _get_llm_client,
)
from backend.services.character_extractor import CharacterExtractor
from backend.services.relationship_extractor import RelationshipExtractor
from backend.services.visualization import generate_relationship_graph_html


def ensure_results_dir() -> None:
    """Ensure the results directory exists."""
    os.makedirs(config.nlp.out_dir, exist_ok=True)


def out_path(filename: str) -> str:
    """Get the full path to a results file."""
    return os.path.join(config.nlp.out_dir, filename)


def main(pdf_path: str):
    """Main CLI entry point."""
    ensure_results_dir()

    out_chars = out_path("characters.json")
    out_rels = out_path("relationships.jsonl")
    out_rel_graph = out_path("character_graph.html")

    print("\n" + "=" * 60)
    print("CHARACTER RELATIONSHIP EXTRACTION")
    print("=" * 60)
    print(f"Input: {pdf_path}")
    print(f"Output: {config.nlp.out_dir}/")

    # Build NLP pipeline
    nlp = _build_nlp()

    # Get LLM client
    client = _get_llm_client()

    # Extract text
    text = _extract_text_from_pdf(pdf_path, config.nlp.max_pages)

    doc = nlp(text)
    sentences = [
        s.text.strip()
        for s in doc.sents
        if len(s.text.strip()) >= config.nlp.min_sentence_len
    ]
    print(f"[INFO] Sentences: {len(sentences)}")

    # Module 1: Character Extraction
    char_ext = CharacterExtractor(nlp, min_person_freq=config.nlp.min_person_freq)
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

    # Module 2: Relationship Extraction
    rel_config = {
        "min_sentence_len": config.nlp.min_sentence_len,
        "window_sentences": config.nlp.window_sentences,
        "window_stride": config.nlp.window_stride,
        "max_context_chars": config.nlp.max_context_chars,
        "max_people_per_window": config.nlp.max_people_per_window,
        "min_interaction_weight": config.nlp.min_interaction_weight,
        "evidence_per_pair": config.nlp.evidence_per_pair,
        "max_pairs_to_describe": config.nlp.max_pairs_to_describe,
        "min_quote_words": config.nlp.min_quote_words,
        "max_quote_words": config.nlp.max_quote_words,
        "require_quotes": config.nlp.require_quotes,
        "model": config.llm.local_model if config.llm.use_local else config.llm.openai_model,
        "temperature": config.llm.temperature,
        "max_workers": config.llm.max_workers,
        "min_request_interval_sec": config.llm.min_request_interval_sec,
        "max_retries": config.llm.max_retries,
    }

    rel_ext = RelationshipExtractor(nlp, char_ext, client, rel_config)
    relationships = rel_ext.extract(text)

    if not relationships:
        print("[ERROR] No relationships extracted.")
        sys.exit(3)

    with open(out_rels, "w", encoding="utf-8") as f:
        for r in relationships:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE] Saved: {out_rels}")

    # Generate visualization
    print("\n[INFO] Generating graph...")
    rel_graph = rel_ext.create_graph()
    graph_html = generate_relationship_graph_html(rel_graph)

    with open(out_rel_graph, "w", encoding="utf-8") as f:
        f.write(graph_html)
    print(f"[DONE] Saved: {out_rel_graph}")

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
  python cli.py books/novel.pdf
  python cli.py books/novel.pdf --local_model mistral

  # Cloud mode (requires OPENAI_API_KEY)
  python cli.py books/novel.pdf --cloud

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
        default=config.llm.local_model,
        help="Ollama model (default: llama3.2)",
    )
    ap.add_argument(
        "--model", default=config.llm.openai_model, help="OpenAI model (for --cloud)"
    )
    ap.add_argument("--min_person_freq", type=int, default=config.nlp.min_person_freq)
    ap.add_argument("--max_pages", type=int, default=None)
    args = ap.parse_args()

    # Update config based on CLI args
    config.llm.use_local = not args.cloud
    config.llm.local_model = args.local_model
    config.llm.openai_model = args.model
    config.nlp.min_person_freq = args.min_person_freq
    if args.max_pages:
        config.nlp.max_pages = args.max_pages

    if not config.llm.use_local and not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY required for --cloud mode")
        print("Use --local for free local inference with Ollama")
        sys.exit(1)

    main(args.pdf)
