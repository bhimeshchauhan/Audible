"""
Graph visualization generation.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List

import networkx as nx

from .text_utils import CATEGORY_COLORS, log_score


def generate_relationship_graph_html(
    G: nx.DiGraph, title: str = "Character Relationships"
) -> str:
    """
    Generate relationship graph HTML as a string.

    Args:
        G: NetworkX directed graph with relationship data
        title: Title for the visualization

    Returns:
        Complete HTML string with embedded vis.js visualization
    """
    node_weights: Counter[str] = Counter()
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        node_weights[u] += w
        node_weights[v] += w

    nodes_data: List[Dict] = [
        {"id": n, "label": n, "value": 10 + 4 * log_score(node_weights.get(n, 1))}
        for n in G.nodes()
    ]

    edges_data: List[Dict] = []
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
<div class="header"><h1>{title}</h1></div>
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

    return html
