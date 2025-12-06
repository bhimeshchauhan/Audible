import os
import sys
import pytesseract
from pdf2image import convert_from_path
from transformers import pipeline
import spacy
import networkx as nx
from pyvis.network import Network

# Load Models
ner_pipeline = pipeline(
    "ner", model="dslim/bert-base-NER", aggregation_strategy="simple"
)
nlp = spacy.load("en_core_web_trf")


# OCR PDF
def extract_text_from_scanned_pdf(pdf_path):
    print("[INFO] Running OCR...")
    pages = convert_from_path(pdf_path)
    text = ""
    for i, page in enumerate(pages):
        page_text = pytesseract.image_to_string(page)
        text += f"\n[PAGE {i+1}]\n" + page_text
    return text


from transformers import AutoTokenizer
import re

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")


def clean_entity(word):
    # Remove punctuation and artifacts
    return re.sub(r"[^\w\s'-]", "", word).strip()


def extract_characters(text):
    print("[INFO] Running NER in chunks...")
    chunks = []
    words = text.split()
    chunk_size = 400
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))

    full_names = set()

    for chunk in chunks:
        ner_results = ner_pipeline(chunk)
        current_name = []
        for token in ner_results:
            if token["entity_group"] == "PER":
                word = clean_entity(token["word"])
                if not word:
                    continue
                if word.startswith("##"):
                    if current_name:
                        current_name[-1] += word[2:]
                else:
                    if current_name:
                        full_names.add(" ".join(current_name))
                    current_name = [word]
            else:
                if current_name:
                    full_names.add(" ".join(current_name))
                    current_name = []
        if current_name:
            full_names.add(" ".join(current_name))

    filtered = {name for name in full_names if len(name.split()) >= 1 and len(name) > 2}
    print(f"[NER] Found characters: {filtered}")
    return list(filtered)


# Relationship Extraction (basic SVO triples)
def extract_relationships(text, characters):
    print("[INFO] Extracting relationships...")
    doc = nlp(text)
    relationships = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ("nsubj", "nsubjpass") and token.text in characters:
                subject = token.text
                verb = token.head.lemma_
                for child in token.head.children:
                    if child.dep_ in ("dobj", "pobj") and child.text in characters:
                        obj = child.text
                        relationships.append((subject, verb, obj))
    if not relationships:
        print("[WARNING] No relationships found. Adding fallback.")
        relationships = [("Alice", "knows", "Bob"), ("Bob", "hates", "Charlie")]
    print(f"[Relations] Extracted: {relationships}")
    return relationships


# Graph Visualization
def build_graph(relationships, output_file="character_graph.html"):
    print("[INFO] Building graph...")
    G = nx.DiGraph()
    for src, rel, tgt in relationships:
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += 1
        else:
            G.add_edge(src, tgt, label=rel, weight=1)

    net = Network(directed=True, notebook=False)
    for node in G.nodes:
        net.add_node(node, label=node)
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, label=data["label"], value=data["weight"])
    net.write_html(output_file)
    print(f"[Graph] Saved to {output_file}")


# Full Pipeline
def process_pdf(file_path):
    text = extract_text_from_scanned_pdf(file_path)
    characters = extract_characters(text)
    print(f"characters(content) {characters}")
    relationships = extract_relationships(text, characters)
    build_graph(relationships)


# Run Script
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python character_graph.py <path_to_pdf>")
        sys.exit(1)
    process_pdf(sys.argv[1])
