"""
prepare_training_examples.py (HGT Upgraded)
-------------------------------------------
Translates zoom.csv (human-readable attack paths) into
training_examples.json containing both Heterogeneous coordinate pairs
and legacy global indices.

python scripts/prepare_training_examples.py \
    --graph data_work/training_data/testing/output/forest_graph.json \
    --zoom data_work/training_data/testing/zoom.csv \
    --out data_work/training_data/testing/output/training_examples.json \
    --hetero data_work/training_data/testing/output/heterodata.pt

"""

import json
import csv
import argparse
import sys
from pathlib import Path
from collections import defaultdict

NODE_TYPES = ["users", "computers", "groups", "domains",
              "gpos", "ous", "containers", "cas", "certtemplates"]


# ── name → index resolver (Hetero Upgrade) ──────────────────────────────────

def build_resolver(graph_path: Path) -> tuple[dict, dict, dict, int, int]:
    """
    Builds the heterogeneous coordinate map.
    Returns: (global_map, name_to_sid, sid_to_global, sentinel_local, sentinel_global)
    """
    forest = json.loads(graph_path.read_text(encoding="utf-8"))

    global_map    = {}  # SID -> (node_type, local_idx)
    name_to_sid   = {}
    sid_to_global = {}  # SID -> flat global index
    
    global_counter = 0

    for nt in NODE_TYPES:
        local_idx = 0
        for obj in forest.get(nt, []) or []:
            oid = (obj.get("ObjectIdentifier") or "").upper()
            if not oid or oid in global_map:
                continue

            name = (obj.get("Properties", {}).get("name") or "").lower()
            
            global_map[oid]    = (nt, local_idx)
            sid_to_global[oid] = global_counter
            name_to_sid[name]  = oid

            short = name.split("@")[0].split(".")[0]
            if short and short not in name_to_sid:
                name_to_sid[short] = oid

            local_idx += 1
            global_counter += 1

    # The null sentinel is dynamically attached to the end of the "users" tensor
    sentinel_local  = len(forest.get("users", []))
    sentinel_global = global_counter

    return global_map, name_to_sid, sid_to_global, sentinel_local, sentinel_global


def resolve_sid(val: str, name_to_sid: dict, global_map: dict):
    if not val or not val.strip():
        return None
    v = val.strip()
    oid = v.upper() if v.upper() in global_map else None
    
    if not oid:
        lower = v.lower()
        oid = name_to_sid.get(lower)
        if not oid:
            short = lower.split("@")[0].split(".")[0]
            oid = name_to_sid.get(short)
            
    return oid if oid in global_map else None


# ── zoom.csv parsing ────────────────────────────────────────────────────────

def parse_zoom(zoom_path: Path,
               name_to_sid: dict,
               global_map: dict,
               sid_to_global: dict,
               sentinel_local: int,
               sentinel_global: int) -> tuple[list, dict]:
               
    examples = []
    paths    = defaultdict(list)
    raw_rows = []

    with open(zoom_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_rows.append(row)
            paths[row["path_id"].strip()].append(row)

    objectives = set()
    for pid, rows in paths.items():
        rows_sorted  = sorted(rows, key=lambda r: int(r["step"]))
        last_hop_sid = resolve_sid(rows_sorted[-1]["next_hop_id"], name_to_sid, global_map)
        if last_hop_sid:
            objectives.add(last_hop_sid)

    stats = {
        "rows":               len(raw_rows),
        "paths":              len(paths),
        "objectives":         len(objectives),
        "skipped_invisible":  0,
        "skipped_unresolved": 0,
        "constrained":        0,
        "unconstrained":      0,
    }

    for row in raw_rows:
        path_id    = row["path_id"].strip()
        step       = int(row["step"])
        train_on   = int(row.get("train_on_step", "1") or "1")
        path_w     = float(row.get("path_weight", "1.0") or "1.0")
        step_cost  = float(row.get("step_cost",   "0.0") or "0.0")
        eff_weight = path_w * (1.0 - step_cost)

        # Skip logic handles 'Unauthenticated' and name anomalies gracefully
        if train_on == 0:
            stats["skipped_invisible"] += 1
            continue

        cur_sid  = resolve_sid(row["current_node_id"],    name_to_sid, global_map)
        hop_sid  = resolve_sid(row["next_hop_id"],        name_to_sid, global_map)
        cred_sid = resolve_sid(row["credential_used_id"], name_to_sid, global_map)

        if not cur_sid or not hop_sid or not cred_sid:
            unresolved = []
            if not cur_sid:  unresolved.append(f"current='{row['current_node_id']}'")
            if not hop_sid:  unresolved.append(f"next_hop='{row['next_hop_id']}'")
            if not cred_sid: unresolved.append(f"credential='{row['credential_used_id']}'")
            print(f"  WARN [{path_id} step {step}]: unresolved → {', '.join(unresolved)}")
            stats["skipped_unresolved"] += 1
            continue

        cur_type, cur_local = global_map[cur_sid]
        hop_type, hop_local = global_map[hop_sid]

        # Mode 1 & 2: Constrained (real cred) and Unconstrained (null sentinel)
        for mode, c_sid in [("constrained", cred_sid), ("unconstrained", "SENTINEL")]:
            if c_sid == "SENTINEL":
                cred_type, cred_local, cred_global = "users", sentinel_local, sentinel_global
            else:
                cred_type, cred_local = global_map[c_sid]
                cred_global = sid_to_global[c_sid]

            examples.append({
                "path_id":          path_id,
                "step":             step,
                "mode":             mode,
                # New Hetero PyG Coordinates
                "current_type":     cur_type,
                "current_local":    cur_local,
                "credential_type":  cred_type,
                "credential_local": cred_local,
                "next_hop_type":    hop_type,
                "next_hop_local":   hop_local,
                # Legacy Flat IDs (kept for backward compatibility)
                "current_idx":      sid_to_global[cur_sid],
                "credential_idx":   cred_global,
                "next_hop_idx":     sid_to_global[hop_sid],
                "eff_weight":       round(eff_weight, 4),
                "notes":            row.get("notes", ""),
            })
            stats[mode] += 1

    return examples, stats


# ── optional heterodata sanity check ────────────────────────────────────────

def validate_against_hetero(hetero_path: Path, examples: list) -> tuple[bool, str]:
    """
    Verify that the typed local indices in our generated examples do not
    exceed the bounds of the actual PyG HeteroData tensors.
    """
    try:
        import torch
        data = torch.load(hetero_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, f"could not load heterodata: {e}"

    bad = []
    for e in examples:
        # Check Current
        ct = e["current_type"]
        if ct not in data.node_types or e["current_local"] >= data[ct].num_nodes:
            bad.append((e["path_id"], e["step"], "current", ct, e["current_local"]))
            
        # Check Hop
        ht = e["next_hop_type"]
        if ht not in data.node_types or e["next_hop_local"] >= data[ht].num_nodes:
            bad.append((e["path_id"], e["step"], "next_hop", ht, e["next_hop_local"]))
            
        # Check Credential
        crt = e["credential_type"]
        # Allow +1 for the sentinel appended to the 'users' tensor
        allowance = 1 if crt == "users" else 0
        if crt not in data.node_types or e["credential_local"] >= (data[crt].num_nodes + allowance):
            bad.append((e["path_id"], e["step"], "credential", crt, e["credential_local"]))

    if bad:
        return False, f"{len(bad)} out-of-range typed indices found (showing 3): {bad[:3]}"

    types_str = ", ".join([f"{nt}:{data[nt].num_nodes}" for nt in data.node_types])
    return True, f"validated against heterodata boundaries ({types_str})"


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
                        help="Optional heterodata.pt for tensor boundary validation")
    args = parser.parse_args()

    graph_path = Path(args.graph)
    zoom_path  = Path(args.zoom)
    out_path   = Path(args.out)

    print("=" * 60)
    print("prepare_training_examples.py (HGT Upgraded)")
    print("=" * 60)
    print(f"Graph : {graph_path}")
    print(f"Zoom  : {zoom_path}")
    print(f"Out   : {out_path}")
    if args.hetero:
        print(f"Check : {args.hetero}")
    print()

    print("[1] Building heterogeneous resolver from forest_graph...")
    global_map, name_to_sid, sid_to_global, s_local, s_global = build_resolver(graph_path)
    print(f"  Real nodes      : {len(global_map)}")
    print(f"  Null sentinel   : type 'users', local index {s_local}")
    print(f"  Resolvable names: {len(name_to_sid)}")

    print(f"\n[2] Parsing zoom.csv...")
    examples, stats = parse_zoom(zoom_path, name_to_sid, global_map, sid_to_global, s_local, s_global)

    print(f"\n  Rows in CSV          : {stats['rows']}")
    print(f"  Distinct paths       : {stats['paths']}")
    print(f"  Objectives           : {stats['objectives']}")
    print(f"  Skipped (invisible)  : {stats['skipped_invisible']}")
    print(f"  Skipped (unresolved) : {stats['skipped_unresolved']}")
    print(f"  Constrained examples : {stats['constrained']}")
    print(f"  Unconstrained twins  : {stats['unconstrained']}")
    print(f"  Total examples       : {len(examples)}")

    if args.hetero:
        print(f"\n[3] Validating coordinate boundaries against heterodata...")
        ok, msg = validate_against_hetero(Path(args.hetero), examples)
        if ok:
            print(f"  ✓ {msg}")
        else:
            print(f"  ✗ {msg}", file=sys.stderr)
            return 1

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