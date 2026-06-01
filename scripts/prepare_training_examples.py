"""
prepare_training_examples.py
----------------------------
Translates zoom.csv (human-readable attack paths) into
training_examples.json (indexed tuples consumed by the trainer).

This is a separate step from build_dataset.py because it's only
needed for training. Inference doesn't touch it.

INPUTS
    --graph    forest_graph.json    (for name → index resolution)
    --zoom     zoom.csv             (expert traces)
    --hetero   heterodata.pt        (optional, only used to validate
                                      that indices match the tensor's
                                      node count, including the null sentinel)

OUTPUTS
    training_examples.json

zoom.csv expected columns:
    path_id, step, current_node_id, next_hop_id, credential_used_id,
    path_weight, train_on_step, notes, step_cost

Each row produces two examples (Mode 1 + Mode 2 twin):
    constrained   — uses real credential_idx
    unconstrained — uses null sentinel (last node index)

eff_weight = path_weight * (1 - step_cost)
"""

import json
import csv
import argparse
import sys
from pathlib import Path
from collections import defaultdict

NODE_TYPES = ["users", "computers", "groups", "domains",
              "gpos", "ous", "containers", "cas", "certtemplates"]


# ── name → index resolver (mirrors build_dataset.py ordering) ───────────────

def build_resolver(graph_path: Path) -> tuple[dict, dict, int]:
    """
    Walk forest_graph.json in the same order as build_dataset.py
    to recover the exact (oid → idx) and (name → idx) mappings.
    Returns (id_to_idx, name_to_idx, n_real_nodes).
    """
    forest = json.loads(graph_path.read_text(encoding="utf-8"))

    id_to_idx   = {}
    name_to_idx = {}

    for nt in NODE_TYPES:
        for obj in forest.get(nt, []) or []:
            oid = (obj.get("ObjectIdentifier") or "").upper()
            if not oid or oid in id_to_idx:
                continue

            name = (obj.get("Properties", {}).get("name") or "").lower()
            idx  = len(id_to_idx)
            id_to_idx[oid]    = idx
            name_to_idx[name] = idx

            # short name (before @ or first dot)
            short = name.split("@")[0].split(".")[0]
            if short and short not in name_to_idx:
                name_to_idx[short] = idx

    return id_to_idx, name_to_idx, len(id_to_idx)


def resolve(val: str, name_to_idx: dict, id_to_idx: dict):
    if not val or not val.strip():
        return None
    v = val.strip()
    if v.upper() in id_to_idx:
        return id_to_idx[v.upper()]
    lower = v.lower()
    if lower in name_to_idx:
        return name_to_idx[lower]
    short = lower.split("@")[0].split(".")[0]
    if short in name_to_idx:
        return name_to_idx[short]
    return None


# ── zoom.csv parsing ────────────────────────────────────────────────────────

def parse_zoom(zoom_path: Path,
               name_to_idx: dict,
               id_to_idx: dict,
               null_idx: int) -> tuple[list, dict]:
    """
    Returns (examples_list, stats_dict).
    Each constrained row becomes 2 examples: constrained + unconstrained twin.
    """
    examples = []
    paths    = defaultdict(list)
    raw_rows = []

    with open(zoom_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_rows.append(row)
            paths[row["path_id"].strip()].append(row)

    objectives = set()
    for pid, rows in paths.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["step"]))
        last_hop    = rows_sorted[-1]["next_hop_id"].strip().upper()
        idx         = resolve(last_hop, name_to_idx, id_to_idx)
        if idx is not None:
            objectives.add(idx)

    stats = {
        "rows":            len(raw_rows),
        "paths":           len(paths),
        "objectives":      len(objectives),
        "skipped_invisible": 0,
        "skipped_unresolved": 0,
        "constrained":     0,
        "unconstrained":   0,
    }

    for row in raw_rows:
        path_id    = row["path_id"].strip()
        step       = int(row["step"])
        train_on   = int(row.get("train_on_step", "1") or "1")
        path_w     = float(row.get("path_weight", "1.0") or "1.0")
        step_cost  = float(row.get("step_cost",   "0.0") or "0.0")
        eff_weight = path_w * (1.0 - step_cost)

        if train_on == 0:
            stats["skipped_invisible"] += 1
            continue

        cur  = resolve(row["current_node_id"],    name_to_idx, id_to_idx)
        hop  = resolve(row["next_hop_id"],        name_to_idx, id_to_idx)
        cred = resolve(row["credential_used_id"], name_to_idx, id_to_idx)

        if cur is None or hop is None or cred is None:
            unresolved = []
            if cur  is None: unresolved.append(f"current='{row['current_node_id']}'")
            if hop  is None: unresolved.append(f"next_hop='{row['next_hop_id']}'")
            if cred is None: unresolved.append(f"credential='{row['credential_used_id']}'")
            print(f"  WARN [{path_id} step {step}]: unresolved → {', '.join(unresolved)}")
            stats["skipped_unresolved"] += 1
            continue

        # Mode 1 — constrained (real credential)
        examples.append({
            "path_id":        path_id,
            "step":           step,
            "current_idx":    cur,
            "credential_idx": cred,
            "next_hop_idx":   hop,
            "eff_weight":     round(eff_weight, 4),
            "mode":           "constrained",
            "notes":          row.get("notes", ""),
        })
        stats["constrained"] += 1

        # Mode 2 — unconstrained twin (null credential sentinel)
        examples.append({
            "path_id":        path_id,
            "step":           step,
            "current_idx":    cur,
            "credential_idx": null_idx,
            "next_hop_idx":   hop,
            "eff_weight":     round(eff_weight, 4),
            "mode":           "unconstrained",
            "notes":          row.get("notes", ""),
        })
        stats["unconstrained"] += 1

    return examples, stats


# ── optional heterodata sanity check ────────────────────────────────────────

def validate_against_hetero(hetero_path: Path,
                            examples: list,
                            n_real_nodes: int) -> tuple[bool, str]:
    """
    Verify that the indices in our generated examples are valid for the
    heterodata's actual node count. Returns (ok, message).
    """
    try:
        import torch
        data = torch.load(hetero_path, map_location="cpu", weights_only=False)
        n_hetero = data["node"].num_nodes
    except Exception as e:
        return False, f"could not load heterodata: {e}"

    # heterodata includes the null sentinel = n_real_nodes + 1
    expected = n_real_nodes + 1
    if n_hetero != expected:
        return False, (f"node count mismatch — heterodata has {n_hetero} nodes, "
                       f"expected {expected} ({n_real_nodes} real + 1 sentinel). "
                       f"Did the graph change since heterodata was built?")

    bad = []
    for e in examples:
        for key in ["current_idx", "credential_idx", "next_hop_idx"]:
            if not (0 <= e[key] < n_hetero):
                bad.append((e["path_id"], e["step"], key, e[key]))

    if bad:
        return False, f"{len(bad)} out-of-range indices found (showing 3): {bad[:3]}"

    return True, f"validated against heterodata ({n_hetero} nodes)"


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph",  required=True,
                        help="Path to forest_graph.json (for index resolution)")
    parser.add_argument("--zoom",   required=True,
                        help="Path to zoom.csv (expert traces)")
    parser.add_argument("--out",    required=True,
                        help="Output training_examples.json")
    parser.add_argument("--hetero", default=None,
                        help="Optional heterodata.pt for index validation")
    args = parser.parse_args()

    graph_path = Path(args.graph)
    zoom_path  = Path(args.zoom)
    out_path   = Path(args.out)

    print("=" * 60)
    print("prepare_training_examples.py")
    print("=" * 60)
    print(f"Graph : {graph_path}")
    print(f"Zoom  : {zoom_path}")
    print(f"Out   : {out_path}")
    if args.hetero:
        print(f"Check : {args.hetero}")
    print()

    # ─ build resolver ─
    print("[1] Building resolver from forest_graph...")
    id_to_idx, name_to_idx, n_real_nodes = build_resolver(graph_path)
    null_idx = n_real_nodes  # sentinel = n_real_nodes (build_dataset adds zero row at this index)
    print(f"  Real nodes      : {n_real_nodes}")
    print(f"  Null sentinel at: {null_idx}")
    print(f"  Resolvable names: {len(name_to_idx)}")

    # ─ parse zoom.csv ─
    print(f"\n[2] Parsing zoom.csv...")
    examples, stats = parse_zoom(zoom_path, name_to_idx, id_to_idx, null_idx)

    print(f"\n  Rows in CSV       : {stats['rows']}")
    print(f"  Distinct paths    : {stats['paths']}")
    print(f"  Objectives        : {stats['objectives']}")
    print(f"  Skipped (invisible): {stats['skipped_invisible']}")
    print(f"  Skipped (unresolved): {stats['skipped_unresolved']}")
    print(f"  Constrained examples : {stats['constrained']}")
    print(f"  Unconstrained twins  : {stats['unconstrained']}")
    print(f"  Total examples       : {len(examples)}")

    # ─ optional heterodata validation ─
    if args.hetero:
        print(f"\n[3] Validating against heterodata...")
        ok, msg = validate_against_hetero(Path(args.hetero), examples, n_real_nodes)
        if ok:
            print(f"  ✓ {msg}")
        else:
            print(f"  ✗ {msg}", file=sys.stderr)
            return 1

    # ─ write output ─
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(examples, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nWritten {out_path}")
    print(f"  Total examples: {len(examples)}")
    print("\nNext: upload heterodata.pt + training_examples.json to Kaggle and train")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())