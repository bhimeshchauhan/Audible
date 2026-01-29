"""
Main PDF processing service that orchestrates character and relationship extraction.
"""

from __future__ import annotations

import os
from typing import Optional

import PyPDF2
import spacy
from openai import OpenAI

from ..config import config
from .character_extractor import CharacterExtractor
from .relationship_extractor import RelationshipExtractor
from .text_utils import clean_text
from .visualization import generate_relationship_graph_html

# Lazy-loaded NLP pipeline
_nlp = None


def _get_nlp():
    """Get or create the NLP pipeline (singleton)."""
    global _nlp
    if _nlp is None:
        _nlp = _build_nlp()
    return _nlp


def _build_nlp():
    """Build the spaCy NLP pipeline with optional coreference resolution."""
    from fastcoref import spacy_component  # noqa: F401 - registers "fastcoref"

    nlp = spacy.load(config.nlp.spacy_model)
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)
    if config.nlp.enable_coref and "fastcoref" not in nlp.pipe_names:
        nlp.add_pipe(
            "fastcoref", config={"model_architecture": config.nlp.coref_model_arch}
        )
    return nlp


def _extract_text_from_pdf(pdf_path: str, max_pages: Optional[int]) -> str:
    """Extract text content from a PDF file."""
    print("[INFO] Extracting text from PDF...")
    out: list[str] = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages = reader.pages if max_pages is None else reader.pages[:max_pages]
        for page in pages:
            t = page.extract_text() or ""
            if t.strip():
                out.append(t)
    return clean_text("\n".join(out))


def _get_llm_client() -> OpenAI:
    """Create and return the appropriate LLM client."""
    if config.llm.use_local:
        print(f"[INFO] Using LOCAL LLM: {config.llm.local_model} via Ollama")
        return OpenAI(base_url=config.llm.local_base_url, api_key="ollama")
    else:
        print(f"[INFO] Using CLOUD LLM: {config.llm.openai_model} via OpenAI")
        return OpenAI()


def process_pdf_to_graph_html(
    pdf_path: str, max_pages: Optional[int] = None
) -> str:
    """
    Process a PDF file and return the character relationship graph as HTML.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Optional maximum number of pages to process

    Returns:
        HTML string containing the interactive relationship graph

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If insufficient characters or relationships are found
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print("\n" + "=" * 60)
    print("CHARACTER RELATIONSHIP EXTRACTION (API)")
    print("=" * 60)
    print(f"Input: {pdf_path}")

    # Build NLP pipeline
    nlp = _get_nlp()

    # Set up LLM client
    client = _get_llm_client()

    # Use config max_pages if not specified
    effective_max_pages = max_pages if max_pages is not None else config.nlp.max_pages

    # Extract text
    text = _extract_text_from_pdf(pdf_path, effective_max_pages)

    if not text.strip():
        raise ValueError("No text could be extracted from the PDF")

    doc = nlp(text)
    sentences = [
        s.text.strip()
        for s in doc.sents
        if len(s.text.strip()) >= config.nlp.min_sentence_len
    ]
    print(f"[INFO] Sentences: {len(sentences)}")

    if len(sentences) < 5:
        raise ValueError("PDF contains too few sentences for meaningful analysis")

    # Module 1: Character Extraction
    char_ext = CharacterExtractor(nlp, min_person_freq=config.nlp.min_person_freq)
    char_result = char_ext.extract(sentences)

    if len(char_result["characters"]) < 2:
        raise ValueError(
            f"Found only {len(char_result['characters'])} character(s). "
            "Need at least 2 characters for relationship extraction."
        )

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
        raise ValueError(
            "No relationships could be extracted between characters. "
            "The PDF may not contain sufficient narrative content."
        )

    # Generate graph
    print("\n[INFO] Generating relationship graph...")
    rel_graph = rel_ext.create_graph()
    graph_html = generate_relationship_graph_html(rel_graph)

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Characters: {len(char_result['characters'])}")
    print(f"  Relationships: {len(relationships)}")

    return graph_html
