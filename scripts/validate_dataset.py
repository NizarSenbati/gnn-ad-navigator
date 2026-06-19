"""
validate_dataset.py (HGT Upgrade + ADCS Focus)
----------------------------------------------
Pre-training audit of the build_dataset.py HeteroData output. 
Run this before training to catch data issues that would burn Kaggle quota.
"""

import json
import argparse
import sys
import torch
from pathlib import Path
from collections import Counter, defaultdict

def check(condition: bool, msg: str, warnings: list, fatal: bool = False) -> None:
    if not condition:
        marker = "✗ FAIL" if fatal else "⚠ WARN"
        warnings.append((fatal, msg))
        print(f"  {marker}  {msg}")
    else:
        print(f"  ✓ OK    {msg}")

# ── individual checks ─────────────────────────────────────────────────────────

def check_node_features(data, warnings):
    print("\n[ Node features (Heterogeneous) ]")
    
    check(len(data.node_types) > 0, "At least one node type present", warnings, fatal=True)
    
    for nt in data.node_types:
        x = data[nt].x
        n, d = x.shape
        
        # Check for NaN / inf
        has_nan = torch.isnan(x).any().item()
        has_inf = torch.isinf(x).any().item()
        check(not has_nan, f"{nt}: No NaN values", warnings, fatal=True)
        check(not has_inf, f"{nt}: No Inf values", warnings, fatal=True)

        if n == 0:
            continue
            
        nonzero_ratio = (x != 0).float().mean().item()
        print(f"          {nt:<15} : {n:<5} nodes | {d} features | {100*nonzero_ratio:>4.1f}% density")


def check_edges(data, warnings):
    print("\n[ Edges (Triplets) ]")
    edge_types = data.edge_types
    check(len(edge_types) > 0, "At least one edge type present", warnings, fatal=True)

    total_edges = 0
    self_loops  = 0
    edge_counts = []

    for triplet in edge_types:
        ei = data[triplet].edge_index
        n  = ei.shape[1]
        total_edges += n
        edge_counts.append((triplet, n))

        # count self-loops (only if source and dest types are the same)
        if triplet[0] == triplet[2]:
            sl = (ei[0] == ei[1]).sum().item()
            self_loops += sl

    print(f"          Edge types         : {len(edge_types)}")
    print(f"          Total edge entries : {total_edges}")
    
    check(total_edges > 10, f"Edge total ({total_edges}) is non-trivial", warnings)
    check(self_loops == 0, f"No self-loops (got {self_loops})", warnings)

    # Print top 5 edge types
    edge_counts.sort(key=lambda x: -x[1])
    for triplet, count in edge_counts[:5]:
        src, rel, dst = triplet
        print(f"          Top edge: {rel:<20} ({src} → {dst}) : {count}")


def check_adcs_health(data, warnings):
    """Specific ADCS checks to ensure Certipy data survived the pipeline."""
    print("\n[ ADCS Health Check ]")
    
    # 1. Check Node Existence
    has_cas = "cas" in data.node_types and data["cas"].num_nodes > 0
    has_tpl = "certtemplates" in data.node_types and data["certtemplates"].num_nodes > 0
    
    check(has_cas, f"CA nodes present ({data['cas'].num_nodes if 'cas' in data.node_types else 0})", warnings)
    check(has_tpl, f"Template nodes present ({data['certtemplates'].num_nodes if 'certtemplates' in data.node_types else 0})", warnings)

    # 2. Check Edge Existence
    adcs_rels = {"enroll", "publishedto", "writeowner", "writedacl", "genericall", "trustedforntauth"}
    found_rels = {triplet[1].lower() for triplet in data.edge_types}
    
    intersect = adcs_rels.intersection(found_rels)
    missing   = adcs_rels - found_rels
    
    if "enroll" in found_rels:
        check(True, "Critical ADCS path 'Enroll' is intact", warnings)
    else:
        check(False, "Missing 'Enroll' edges (ADCS pathfinding will fail)", warnings)
        
    if "trustedforntauth" in found_rels:
        check(True, "Critical CA escape route 'TrustedForNTAuth' is intact", warnings)
    else:
        check(False, "Missing 'TrustedForNTAuth' edges (CA black hole bug)", warnings)

    print(f"          ADCS edge types detected : {', '.join(intersect) if intersect else 'None'}")


def check_training_examples(examples, data, warnings):
    print("\n[ Training examples ]")
    check(len(examples) > 0, "Examples list non-empty", warnings, fatal=True)

    n_constrained   = sum(1 for e in examples if e.get("mode") == "constrained")
    n_unconstrained = sum(1 for e in examples if e.get("mode") == "unconstrained")
    print(f"          Constrained        : {n_constrained}")
    print(f"          Unconstrained      : {n_unconstrained}")

    check(n_constrained == n_unconstrained, "Constrained matches unconstrained twin count", warnings)

    # Validate HGT Typed Boundaries
    out_of_range = 0
    for e in examples:
        # Check Current
        ct = e["current_type"]
        if ct not in data.node_types or e["current_local"] >= data[ct].num_nodes:
            out_of_range += 1
            
        # Check Hop
        ht = e["next_hop_type"]
        if ht not in data.node_types or e["next_hop_local"] >= data[ht].num_nodes:
            out_of_range += 1

    check(out_of_range == 0, f"All PyG tensor coordinates are within bounds", warnings, fatal=(out_of_range > 0))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hetero", required=True, help="Path to heterodata.pt")
    parser.add_argument("--examples", default=None, help="Path to training_examples.json")
    parser.add_argument("--graph", default=None, help="Path to forest_graph.json (for name resolution)")
    args = parser.parse_args()

    print("=" * 60)
    print("validate_dataset.py (HGT Upgrade)")
    print("=" * 60)

    warnings = []

    hetero_path = Path(args.hetero)
    print(f"Loading: {hetero_path}")
    try:
        data = torch.load(hetero_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"FATAL: cannot load heterodata: {e}")
        return 1

    # Run checks
    check_node_features(data, warnings)
    check_edges(data, warnings)
    check_adcs_health(data, warnings)

    if args.examples:
        examples_path = Path(args.examples)
        if examples_path.exists():
            print(f"\nLoading: {examples_path}")
            examples = json.loads(examples_path.read_text(encoding="utf-8"))
            check_training_examples(examples, data, warnings)
        else:
            print(f"\n  (examples file not found: {examples_path})")

    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    fatal_count = sum(1 for f, _ in warnings if f)
    warn_count  = sum(1 for f, _ in warnings if not f)

    if fatal_count > 0:
        print(f"  ✗ {fatal_count} fatal issue(s) — do not proceed to training")
        for f, msg in warnings:
            if f: print(f"    {msg}")
        return 1

    if warn_count > 0:
        print(f"  ⚠ {warn_count} warning(s) — review before training:")
        for f, msg in warnings:
            if not f: print(f"    - {msg}")
        print(f"\n  Warnings are informational. Training can proceed.")
        return 0

    print("  ✓ All checks passed. Data is mathematically sound and ready for Kaggle.")
    return 0

if __name__ == "__main__":
    sys.exit(main())