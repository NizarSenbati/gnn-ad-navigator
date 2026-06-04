"""
visualisation.py — graph rendering helpers for the GNN-AD-Navigator GUI

Uses pyvis to draw discovered attack paths as interactive networks.
Styling is restrained — light backgrounds, muted accents, no neon.
"""

import math
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pyvis.network import Network


# ─────────────────────────────────────────────────────────────────────────────
# colour palette — professional, office-app appropriate
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "start":          "#1f4e79",   # deep blue — attacker's foothold
    "intermediate":   "#4a7ba8",   # medium blue — pivot nodes
    "target":         "#7d2f2f",   # muted maroon — objective
    "edge":           "#5a6c7d",   # slate grey — path edges
    "background":     "#fafbfc",   # off-white
    "border":         "#d8dee4",   # light grey
    "text":           "#2c3e50",   # dark navy
}

# node-type-specific iconography (Unicode glyphs so nothing extra to install)
TYPE_GLYPH = {
    "users":         "U",
    "computers":     "C",
    "groups":        "G",
    "domains":       "D",
    "gpos":          "P",
    "ous":           "O",
    "containers":    "+",
    "cas":           "A",
    "certtemplates": "T",
}


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _short_label(name: str, max_len: int = 25) -> str:
    """Compress long fully-qualified names for graph labels."""
    if "@" in name:
        return name.split("@")[0][:max_len]
    return name[:max_len]


def _node_color(position: int, total_steps: int) -> str:
    """Assign colour by position in the path."""
    if position == 0:
        return PALETTE["start"]
    if position == total_steps:
        return PALETTE["target"]
    return PALETTE["intermediate"]


def _node_size(position: int, total_steps: int) -> int:
    """Slightly larger nodes at endpoints."""
    if position == 0 or position == total_steps:
        return 32
    return 24


# ─────────────────────────────────────────────────────────────────────────────
# graph rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_path_graph(path, edge_tensors, idx_to_name, idx_to_type,
                       height_px: int = 400):
    """
    Build and embed an interactive pyvis network for the discovered path.

    The path is drawn linearly with annotated edges. Nodes are colour-coded
    by position (start, intermediate, target) and labelled with their
    short name plus a single-letter type glyph.
    """
    net = Network(
        height=f"{height_px}px",
        width="100%",
        bgcolor=PALETTE["background"],
        font_color=PALETTE["text"],
        directed=True,
    )

    # use hierarchical layout — left-to-right, mirrors attack progression
    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": 200,
          "nodeSpacing": 100
        }
      },
      "nodes": {
        "shape": "dot",
        "font": {
          "size": 14,
          "face": "Helvetica",
          "color": "#2c3e50"
        },
        "borderWidth": 2,
        "borderWidthSelected": 3
      },
      "edges": {
        "color": {
          "color": "#5a6c7d",
          "highlight": "#1f4e79"
        },
        "width": 2,
        "smooth": {
          "type": "cubicBezier",
          "forceDirection": "horizontal"
        },
        "arrows": {
          "to": { "enabled": true, "scaleFactor": 0.8 }
        },
        "font": {
          "size": 11,
          "face": "Helvetica",
          "color": "#2c3e50",
          "background": "rgba(250,251,252,0.9)",
          "strokeWidth": 0,
          "align": "middle"
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true
      },
      "physics": { "enabled": false }
    }
    """)

    n = len(path)

    # add nodes
    for i, node_idx in enumerate(path):
        name      = idx_to_name.get(node_idx, f"idx_{node_idx}")
        node_type = idx_to_type.get(node_idx, "?")
        glyph     = TYPE_GLYPH.get(node_type, "?")

        label = f"[{glyph}] {_short_label(name)}"
        title = (
            f"<div style='font-family: Helvetica; padding: 4px;'>"
            f"<b>{name}</b><br>"
            f"<span style='color:#5a6c7d; font-size:0.9em;'>"
            f"type: {node_type}<br>"
            f"step: {i}"
            f"</span></div>"
        )

        net.add_node(
            node_idx,
            label=label,
            title=title,
            color={
                "background": _node_color(i, n - 1),
                "border":     "#1f2d3d",
                "highlight":  {"background": _node_color(i, n - 1),
                                "border": "#1f2d3d"},
            },
            size=_node_size(i, n - 1),
            level=i,
            font={"color": "white" if i in (0, n - 1) else "#2c3e50"},
        )

    # add edges with technique labels
    from inference import get_edge_type, EDGE_TO_TECHNIQUE
    for i in range(n - 1):
        src, dst = path[i], path[i + 1]
        edge_type = get_edge_type(src, dst, edge_tensors)
        tech, _ = EDGE_TO_TECHNIQUE.get(edge_type.lower(),
                                         ("Unknown edge", "Manual review"))
        net.add_edge(
            src, dst,
            label=edge_type,
            title=f"{edge_type} — {tech}",
            color=PALETTE["edge"],
            width=2.5,
        )

    # write to temp HTML and embed
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html",
                                       delete=False, encoding="utf-8") as f:
        net.write_html(f.name)
        html_path = f.name

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    components.html(html, height=height_px + 20, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# path detail table
# ─────────────────────────────────────────────────────────────────────────────

def render_path_table(path, edge_tensors, idx_to_name, edge_to_technique):
    """
    Show step-by-step path details as a clean table beside the graph.
    """
    from inference import get_edge_type

    rows = []
    for i in range(len(path) - 1):
        src, dst = path[i], path[i + 1]
        edge_type = get_edge_type(src, dst, edge_tensors)
        tech, action = edge_to_technique.get(
            edge_type.lower(), ("Unknown edge", "Manual review")
        )

        src_name = _short_label(idx_to_name.get(src, f"idx_{src}"), 30)
        dst_name = _short_label(idx_to_name.get(dst, f"idx_{dst}"), 30)

        rows.append({
            "Step":      i + 1,
            "From":      src_name,
            "To":        dst_name,
            "Edge":      edge_type,
            "Technique": tech,
        })

    if not rows:
        st.info("Single-node path — no transitions to display.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Step":      st.column_config.NumberColumn(width="small"),
            "From":      st.column_config.TextColumn(width="medium"),
            "To":        st.column_config.TextColumn(width="medium"),
            "Edge":      st.column_config.TextColumn(width="small"),
            "Technique": st.column_config.TextColumn(width="medium"),
        }
    )