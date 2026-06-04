"""
app.py — GNN-AD-Navigator graphical interface

Full operator workflow: pick input/output folders, the GUI runs the
prep pipeline (if needed), loads artifacts, and runs path queries.
Mirrors what launch.sh does on the CLI side.

Run from the gnn-ad-navigator/ root:
    streamlit run gui/app.py
"""

import sys
import os
import subprocess
import math
import json
from pathlib import Path
from datetime import datetime

# make scripts/ importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import streamlit as st
import torch

from inference import (
    build_name_lookup,
    find_node,
    beam_search,
    audit_path,
    explain_terminal_state,
    load_gcn,
    load_hgt,
    EDGE_TO_TECHNIQUE,
)

from visualisation import render_path_graph, render_path_table


# ─────────────────────────────────────────────────────────────────────────────
# page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GNN-AD-Navigator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 1rem; }
    h1 { color: #1f2d3d; font-weight: 500; }
    h2, h3 { color: #2c3e50; font-weight: 500; }
    [data-testid="stSidebar"] { background-color: #fafbfc; }
    [data-testid="stMetricValue"] { color: #1f4e79; font-weight: 500; }
    .stButton button {
        background-color: #1f4e79;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 500;
    }
    .stButton button:hover { background-color: #163d5e; }
    hr { border-color: #e1e4e8; margin: 1rem 0; }
    .status-ok    { color: #2d6a4f; font-weight: 500; }
    .status-warn  { color: #b58105; font-weight: 500; }
    .status-fail  { color: #8b2c2c; font-weight: 500; }
    .path-tag {
        background-color: #eef2f6;
        border-radius: 3px;
        padding: 2px 8px;
        font-family: monospace;
        font-size: 0.85em;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# folder picker — tries tkinter, falls back to text input
# ─────────────────────────────────────────────────────────────────────────────

def native_folder_picker(title="Select folder"):
    """Try opening a native folder picker dialog. Returns path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title=title)
        root.destroy()
        return path or None
    except Exception:
        return None


def folder_input(label: str, key: str, default: str,
                  help_text: str = "") -> str:
    """A folder path input with optional 'Browse...' popup."""
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        path = st.text_input(
            label, value=st.session_state.get(key, default),
            key=f"{key}_text", help=help_text,
            label_visibility="visible",
        )
    with col_btn:
        st.markdown("<div style='height:1.6rem;'></div>",
                     unsafe_allow_html=True)
        if st.button("Browse...", key=f"{key}_browse",
                      use_container_width=True):
            picked = native_folder_picker(label)
            if picked:
                st.session_state[key] = picked
                st.rerun()
    st.session_state[key] = path
    return path


# ─────────────────────────────────────────────────────────────────────────────
# pipeline runner — wraps launch.sh
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_prep(input_dir: Path, output_dir: Path,
                       status_container) -> bool:
    """
    Execute launch.sh ./input ./output to prepare artifacts.
    Streams output into the status container.

    Returns True on success, False on failure.
    """
    launch_script = PROJECT_ROOT / "launch.sh"
    if not launch_script.is_file():
        status_container.error(f"launch.sh not found at {launch_script}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with status_container.status(
            "Running data preparation pipeline...", expanded=True
        ) as status:
            proc = subprocess.Popen(
                [str(launch_script), str(input_dir), str(output_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(PROJECT_ROOT),
            )
            lines = []
            for line in proc.stdout:
                lines.append(line.rstrip())
                # show last 20 lines in the live container
                st.code("\n".join(lines[-20:]), language=None)

            proc.wait()

            if proc.returncode != 0:
                status.update(label="Pipeline failed", state="error")
                return False
            status.update(label="Pipeline complete", state="complete")
        return True
    except Exception as e:
        status_container.error(f"Pipeline crashed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# cached resource loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading graph and lookups...")
def load_graph_resources(graph_path: str, hetero_path: str):
    name_to_idx, idx_to_name, idx_to_type = build_name_lookup(Path(graph_path))
    data = torch.load(hetero_path, map_location="cpu", weights_only=False)
    X = data["node"].x
    all_edges = torch.cat(
        [data[et].edge_index.cpu() for et in data.edge_types], dim=1
    )
    edge_index_flat = torch.unique(all_edges, dim=1)
    edge_tensors = {et[1]: data[et].edge_index for et in data.edge_types}
    return {
        "data": data, "X": X,
        "edge_index_flat": edge_index_flat,
        "edge_tensors": edge_tensors,
        "name_to_idx": name_to_idx,
        "idx_to_name": idx_to_name,
        "idx_to_type": idx_to_type,
    }


@st.cache_resource(show_spinner="Loading model...")
def load_model_resources(model_path: str, model_type: str,
                          graph_path: str, hetero_path: str):
    g = load_graph_resources(graph_path, hetero_path)
    device = torch.device("cpu")
    if model_type == "gcn":
        model, emb = load_gcn(Path(model_path), g["X"],
                               g["edge_index_flat"], device)
    else:
        model, emb = load_hgt(Path(model_path), g["X"], g["data"], device)
    return model, emb


# ─────────────────────────────────────────────────────────────────────────────
# header
# ─────────────────────────────────────────────────────────────────────────────

col_title, col_meta = st.columns([3, 1])
with col_title:
    st.title("GNN-AD-Navigator")
    st.markdown(
        "<p style='color:#5a6c7d; font-size:1rem; margin-top:-0.5rem;'>"
        "Active Directory Attack Path Discovery"
        "</p>",
        unsafe_allow_html=True,
    )
with col_meta:
    st.markdown(
        "<p style='text-align:right; color:#5a6c7d; font-size:0.85rem; "
        "margin-top:1rem;'>v1.0 · Master's thesis demo</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Workspace")
    st.caption("Defaults point at the bundled `examples/` folders.")

    default_input  = PROJECT_ROOT / "examples" / "input"
    default_output = PROJECT_ROOT / "examples" / "output"

    input_dir  = folder_input(
        "Input folder", key="input_dir",
        default=str(default_input),
        help_text="Folder containing BloodHound JSON files and optional "
                  "Certipy output. Use Browse... to pick from the filesystem."
    )
    output_dir = folder_input(
        "Output folder", key="output_dir",
        default=str(default_output),
        help_text="Where forest_graph.json, heterodata.pt, and inference "
                  "logs are written."
    )

    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    # check what's prepared already
    graph_file  = output_path / "forest_graph.json"
    hetero_file = output_path / "heterodata.pt"
    artifacts_ready = graph_file.is_file() and hetero_file.is_file()

    if artifacts_ready:
        st.success(f"Artifacts found in {output_path.name}/")
    else:
        st.info(
            f"No prepared data in {output_path.name}/. Pipeline will run "
            "on first query."
        )

    st.markdown("---")
    st.subheader("Model")

    model_type = st.radio(
        "Architecture",
        options=["gcn", "hgt"],
        format_func=lambda x: {"gcn": "GCN (baseline)",
                                "hgt": "HGT (heterogeneous)"}[x],
        horizontal=True,
    )

    default_model_name = "GCN.pt" if model_type == "gcn" else "HGT.pt"
    default_model = PROJECT_ROOT / "models" / default_model_name
    model_path = st.text_input(
        "Checkpoint",
        value=str(default_model),
    )
    model_ok = Path(model_path).is_file()
    if not model_ok:
        st.warning(f"Model file not found at {model_path}")

    st.markdown("---")
    st.subheader("Query")

    start_query  = st.text_input(
        "Start node", value="wley",
        help="Where the attacker begins. Case-insensitive fuzzy match."
    )
    target_query = st.text_input(
        "Target node", value="domain admins@inlanefreight.local",
        help="Goal node. Search may terminate earlier at any DCSync-capable "
             "node on the target's domain."
    )

    col_beam, col_depth = st.columns(2)
    with col_beam:
        beam_width = st.number_input("Beam width", 1, 10, 3)
    with col_depth:
        max_depth = st.number_input("Max depth", 2, 15, 6)

    st.markdown("---")

    # force re-prep checkbox (advanced)
    force_reprep = st.checkbox(
        "Re-run pipeline even if artifacts exist",
        value=False,
        help="By default the pipeline is skipped if forest_graph.json and "
             "heterodata.pt already exist in the output folder."
    )

    run_clicked = st.button(
        "Run inference",
        type="primary",
        use_container_width=True,
        disabled=not (input_path.is_dir() and model_ok),
    )


# ─────────────────────────────────────────────────────────────────────────────
# main panel
# ─────────────────────────────────────────────────────────────────────────────

if not run_clicked:
    st.info(
        "Configure the workspace and query in the sidebar, then click "
        "**Run inference**. The pipeline runs automatically if the output "
        "folder doesn't already contain prepared data."
    )

    with st.expander("How to use this tool"):
        st.markdown("""
        1. **Set the workspace.** Click **Browse...** next to *Input folder*
           and pick the directory holding your BloodHound JSON files (plus
           optional Certipy output). Then pick (or accept) an *Output folder*
           — this is where the GUI will write the prepared graph,
           tensors, and run logs.

        2. **Choose model and target.** GCN is the baseline. HGT uses
           heterogeneous attention. Specify start and target nodes using
           full names or short fragments.

        3. **Click Run inference.** On first run the pipeline executes
           automatically (~30s for a small graph). Subsequent queries on
           the same workspace skip prep and run in seconds.

        4. **Inspect results.** Each discovered path renders as an
           interactive graph and a step-by-step table. The Audit block
           flags reliability issues. The Terminal state advisory explains
           how to convert mid-attack capabilities into the final objective.

        Every inference run produces a log file in the output folder named
        `inference_<timestamp>.log` for later review.
        """)

else:
    # ── verify input ───────────────────────────────────────────────────────
    if not input_path.is_dir():
        st.error(f"Input folder does not exist: {input_path}")
        st.stop()

    # ── prep if needed ─────────────────────────────────────────────────────
    needs_prep = not artifacts_ready or force_reprep
    if needs_prep:
        prep_container = st.container()
        ok = run_pipeline_prep(input_path, output_path, prep_container)
        if not ok:
            st.error("Pipeline preparation failed. Check the output above.")
            st.stop()
        # bust cache so next load reads new files
        load_graph_resources.clear()
        load_model_resources.clear()

    # ── load resources ─────────────────────────────────────────────────────
    g = load_graph_resources(str(graph_file), str(hetero_file))
    model, emb = load_model_resources(
        model_path, model_type, str(graph_file), str(hetero_file)
    )

    # resolve nodes
    s_idx = find_node(start_query,  g["name_to_idx"], g["idx_to_name"])
    t_idx = find_node(target_query, g["name_to_idx"], g["idx_to_name"])

    if s_idx is None:
        st.error(f"Could not resolve start node '{start_query}' in graph.")
        st.stop()
    if t_idx is None:
        st.error(f"Could not resolve target node '{target_query}' in graph.")
        st.stop()

    # ── status row ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Start",  g["idx_to_name"][s_idx][:30] +
               ("…" if len(g["idx_to_name"][s_idx]) > 30 else ""))
    c2.metric("Target", g["idx_to_name"][t_idx][:30] +
               ("…" if len(g["idx_to_name"][t_idx]) > 30 else ""))
    c3.metric("Model",  model_type.upper())
    c4.metric("Nodes",  f"{g['data']['node'].num_nodes:,}")

    st.markdown("---")

    # ── run inference ──────────────────────────────────────────────────────
    with st.spinner("Searching attack paths..."):
        paths = beam_search(
            model, emb, g["edge_tensors"], s_idx, t_idx,
            g["idx_to_name"],
            beam_width=int(beam_width), max_depth=int(max_depth),
        )

    if not paths:
        st.warning("No paths found within the specified depth.")
        st.stop()

    # ── persist run to output folder ───────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_path / f"inference_{timestamp}.log"
    log_lines = [
        f"GNN-AD-Navigator inference run",
        f"Timestamp     : {timestamp}",
        f"Model         : {model_type.upper()} ({model_path})",
        f"Start node    : {g['idx_to_name'][s_idx]}",
        f"Target node   : {g['idx_to_name'][t_idx]}",
        f"Beam / Depth  : {beam_width} / {max_depth}",
        "",
        f"Found {len(paths)} candidate path(s):",
        "",
    ]
    for rank, (score, path) in enumerate(paths, 1):
        log_lines.append(f"Path {rank}: score={math.exp(score):.4f}, "
                          f"steps={len(path)-1}")
        for node in path:
            log_lines.append(f"  → {g['idx_to_name'].get(node, '?')}")
        log_lines.append("")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    # ── results display ────────────────────────────────────────────────────
    st.subheader("Results")
    st.caption(f"Top {len(paths)} attack path candidate(s). "
                f"Run saved to `{log_path.name}`.")

    for rank, (score, path) in enumerate(paths, 1):
        with st.container(border=True):
            prob = math.exp(score)

            c_a, c_b, c_c = st.columns([2, 1, 1])
            c_a.markdown(f"**Path {rank}**")
            c_b.metric("Confidence", f"{prob:.3f}")
            c_c.metric("Steps", len(path) - 1)

            # larger graph, table below in expander
            render_path_graph(
                path,
                g["edge_tensors"],
                g["idx_to_name"],
                g["idx_to_type"],
                height_px=520,
            )

            with st.expander("Step-by-step details"):
                render_path_table(
                    path,
                    g["edge_tensors"],
                    g["idx_to_name"],
                    EDGE_TO_TECHNIQUE,
                )

            # audit
            audit = audit_path(path, t_idx, g["edge_tensors"], g["idx_to_name"])
            if audit:
                st.markdown("**Audit**")
                for level, msg in audit:
                    css_class = {
                        "ok":   "status-ok",
                        "warn": "status-warn",
                        "fail": "status-fail",
                    }[level]
                    icon = {"ok": "✓", "warn": "!", "fail": "✗"}[level]
                    st.markdown(
                        f"<p class='{css_class}'>{icon} {msg}</p>",
                        unsafe_allow_html=True,
                    )

            # advisory
            advisory = explain_terminal_state(
                path[-1], t_idx, g["edge_tensors"], g["idx_to_name"]
            )
            if advisory:
                with st.expander("Terminal state advisory"):
                    st.code(advisory, language=None)