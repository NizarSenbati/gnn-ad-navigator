"""
validate_dataset.py
-------------------
Pre-training audit of the build_dataset.py output. Run this before training
to catch data issues that would otherwise burn Kaggle quota.

INPUTS
    heterodata.pt              — output of build_dataset.py
    training_examples.json     — optional; required for training-mode checks
    forest_graph.json          — optional; enables name resolution in report

OUTPUTS
    stdout report
    exit code 0 if clean, 1 if warnings exist

Usage:
    python validate_dataset.py --hetero results/heterodata.pt \\
                                --examples results/training_examples.json \\
                                --graph results/forest_graph.json
"""

import json
import argparse
import sys
import torch
from pathlib import Path
from collections import Counter, defaultdict


# ── utilities ─────────────────────────────────────────────────────────────────

def load_idx_to_name(graph_path: Path) -> dict:
    """Build idx → name mapping from forest_graph.json for readable output."""
    forest = json.loads(graph_path.read_text(encoding="utf-8"))
    idx_to_name = {}
    node_order = ["users","computers","groups","domains",
                   "gpos","ous","containers","cas","certtemplates"]
    i = 0
    for nt in node_order:
        for obj in forest.get(nt, []) or []:
            oid = (obj.get("ObjectIdentifier") or "").upper()
            if not oid:
                continue
            name = (obj.get("Properties", {}).get("name") or "?").lower()
            idx_to_name[i] = (name, nt)
            i += 1
    return idx_to_name


def check(condition: bool, msg: str, warnings: list, fatal: bool = False) -> None:
    """Record a check result."""
    if not condition:
        marker = "✗ FAIL" if fatal else "⚠ WARN"
        warnings.append((fatal, msg))
        print(f"  {marker}  {msg}")
    else:
        print(f"  ✓ OK    {msg}")


# ── individual checks ─────────────────────────────────────────────────────────

def check_node_features(data, warnings):
    print("\n[ Node features ]")
    x = data["node"].x
    n, d = x.shape

    check(n > 10, f"Node count = {n}", warnings)
    check(d > 0,  f"Feature dim = {d}", warnings, fatal=True)

    # check for NaN / inf
    has_nan = torch.isnan(x).any().item()
    has_inf = torch.isinf(x).any().item()
    check(not has_nan, f"No NaN values in feature matrix", warnings, fatal=True)
    check(not has_inf, f"No Inf values in feature matrix", warnings, fatal=True)

    # value range
    fmin = x.min().item()
    fmax = x.max().item()
    print(f"          Feature value range: [{fmin:.3f}, {fmax:.3f}]")
    check(-5 <= fmin and fmax <= 5,
          "Feature values within reasonable range", warnings)

    # density
    nonzero_ratio = (x != 0).float().mean().item()
    print(f"          Non-zero density   : {100*nonzero_ratio:.1f}%")
    check(nonzero_ratio > 0.05,
          "At least 5% of feature matrix is non-zero", warnings)


def check_edges(data, warnings):
    print("\n[ Edges ]")
    edge_types = data.edge_types
    check(len(edge_types) > 0, f"At least one edge type present", warnings,
          fatal=True)

    total_edges = 0
    self_loops  = 0
    edge_counts = []

    for et in edge_types:
        ei = data[et].edge_index
        n  = ei.shape[1]
        total_edges += n
        edge_counts.append((et[1], n))

        # count self-loops
        sl = (ei[0] == ei[1]).sum().item()
        self_loops += sl

    print(f"          Edge types         : {len(edge_types)}")
    print(f"          Total edge entries : {total_edges}")
    print(f"          Self-loops         : {self_loops}")

    check(total_edges > 100,
          f"Edge total ({total_edges}) appears non-trivial", warnings)
    check(self_loops == 0,
          f"No self-loops (got {self_loops})", warnings)

    # check for trust edges
    edge_type_names = {et[1] for et in edge_types}
    trust_present   = any(n in edge_type_names for n in
                          ["sameforesttrust", "crossforesttrust", "trustedby"])
    check(trust_present or len(edge_types) < 10,
          "Trust edges present (or graph too small to require them)", warnings)

    # check edge balance — no single type dominating
    if edge_counts:
        edge_counts.sort(key=lambda x: -x[1])
        top_rel, top_n = edge_counts[0]
        share = 100 * top_n / total_edges
        print(f"          Top edge type      : {top_rel} ({top_n} = {share:.1f}%)")
        check(share < 80,
              f"No single edge type dominates (top is {share:.1f}%)", warnings)


def check_connectivity(data, idx_to_name, warnings):
    print("\n[ Connectivity ]")
    n_nodes = data["node"].num_nodes

    # build adjacency (undirected, for connectivity check)
    adj = defaultdict(set)
    for et in data.edge_types:
        ei = data[et].edge_index
        for s, d in zip(ei[0].tolist(), ei[1].tolist()):
            adj[s].add(d)
            adj[d].add(s)

    # find connected components via BFS
    visited = [False] * n_nodes
    components = []
    for start in range(n_nodes):
        if visited[start]:
            continue
        comp = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            comp.append(node)
            stack.extend(adj[node])
        components.append(comp)

    components.sort(key=len, reverse=True)

    print(f"          Connected components: {len(components)}")
    print(f"          Largest component  : {len(components[0])} nodes "
          f"({100*len(components[0])/n_nodes:.1f}%)")

    isolated_count = sum(1 for c in components if len(c) == 1)
    print(f"          Isolated nodes     : {isolated_count}")

    check(len(components[0]) > n_nodes * 0.5,
          "Largest component covers >50% of nodes", warnings)
    check(isolated_count < n_nodes * 0.2,
          f"Isolated nodes < 20% of total (got {isolated_count})", warnings)

    # degree stats
    degrees = [len(adj[i]) for i in range(n_nodes)]
    avg_deg = sum(degrees) / max(n_nodes, 1)
    max_deg = max(degrees) if degrees else 0
    print(f"          Avg degree         : {avg_deg:.1f}")
    print(f"          Max degree         : {max_deg}")

    # show top-degree nodes
    if idx_to_name:
        top_deg = sorted(range(n_nodes), key=lambda i: -degrees[i])[:5]
        print(f"          Highest-degree nodes:")
        for idx in top_deg:
            name, nt = idx_to_name.get(idx, ("?", "?"))
            print(f"            [{nt}] {name}: {degrees[idx]}")


def check_domains(data, idx_to_name, warnings):
    print("\n[ Domain coverage ]")
    if not idx_to_name:
        print("  (skipped — no forest_graph.json provided)")
        return

    # extract domain suffix from each node name
    domain_counts = Counter()
    for idx in range(data["node"].num_nodes):
        if idx not in idx_to_name:
            continue
        name, _ = idx_to_name[idx]
        if "@" in name:
            suffix = name.split("@", 1)[1]
            domain_counts[suffix] += 1
        elif "." in name and not name.startswith("s-1-5"):
            domain_counts[name] += 1

    if domain_counts:
        print(f"          Domains detected   : {len(domain_counts)}")
        for dom, n in domain_counts.most_common(10):
            print(f"            {dom:<40} {n} nodes")


def check_training_examples(examples, n_nodes, warnings):
    print("\n[ Training examples ]")
    check(len(examples) > 0, f"Examples list non-empty", warnings, fatal=True)

    n_constrained   = sum(1 for e in examples if e.get("mode") == "constrained")
    n_unconstrained = sum(1 for e in examples if e.get("mode") == "unconstrained")
    print(f"          Constrained        : {n_constrained}")
    print(f"          Unconstrained      : {n_unconstrained}")
    print(f"          Total              : {len(examples)}")

    check(n_constrained == n_unconstrained,
          f"Constrained matches unconstrained twin count", warnings)
    check(n_constrained >= 10,
          f"At least 10 constrained examples", warnings)

    # validate all indices are in bounds
    out_of_range = 0
    for e in examples:
        for key in ["current_idx", "credential_idx", "next_hop_idx"]:
            idx = e.get(key, -1)
            if idx < 0 or idx >= n_nodes:
                out_of_range += 1
                break
    check(out_of_range == 0,
          f"All example indices in range [0, {n_nodes})", warnings,
          fatal=(out_of_range > 0))

    # weight distribution
    weights = [e.get("eff_weight", 1.0) for e in examples]
    if weights:
        print(f"          Weight range       : [{min(weights):.3f}, {max(weights):.3f}]")
        print(f"          Weight mean        : {sum(weights)/len(weights):.3f}")

    # path diversity
    path_ids = set(e.get("path_id", "") for e in examples)
    print(f"          Unique path_ids    : {len(path_ids)}")
    check(len(path_ids) >= 5,
          f"At least 5 distinct attack paths", warnings)

    # objective coverage
    targets = set(e.get("next_hop_idx") for e in examples)
    print(f"          Unique target nodes: {len(targets)}")


def check_cross_domain_coverage(data, edge_tensors, idx_to_name, warnings):
    """
    For each trust edge between two domains, check that other cross-domain
    edges exist between nodes in those two domains. If trust is present but
    no other cross-domain connectivity exists, collection may be incomplete.
    """
    import torch

    print("\n[ Cross-domain coverage ]")

    # extract domain string from each node name once
    def domain_of(idx):
        name, _ = idx_to_name.get(idx, ("", ""))
        if "@" in name:
            return name.split("@", 1)[1]
        if "." in name and not name.startswith("s-"):
            return name
        return ""

    # build per-node domain tag array
    n_nodes = data["node"].num_nodes
    node_domain = [domain_of(i) for i in range(n_nodes)]

    # collect trust pairs
    trust_pairs = set()
    for rel in ["sameforesttrust", "crossforesttrust", "trustedby"]:
        ei = edge_tensors.get(rel)
        if ei is None: continue
        for s, d in zip(ei[0].tolist(), ei[1].tolist()):
            sd = node_domain[s]
            dd = node_domain[d]
            if sd and dd and sd != dd:
                trust_pairs.add(tuple(sorted([sd, dd])))

    if not trust_pairs:
        print("  (no trust edges detected)")
        return

    # for each non-trust edge, count cross-domain occurrences
    cross_counts = {pair: 0 for pair in trust_pairs}
    for rel, ei in edge_tensors.items():
        if rel.endswith("trust"): continue
        src_arr = ei[0].tolist()
        dst_arr = ei[1].tolist()
        for s, d in zip(src_arr, dst_arr):
            sd = node_domain[s]
            dd = node_domain[d]
            if not sd or not dd or sd == dd:
                continue
            pair = tuple(sorted([sd, dd]))
            if pair in cross_counts:
                cross_counts[pair] += 1

    for pair, count in cross_counts.items():
        domA, domB = pair
        if count == 0:
            print(f"  ⚠ Trust {domA} ↔ {domB} present but ZERO other "
                  f"cross-domain edges. Collection may be incomplete.")
            warnings.append((False,
                f"Trust between {domA} and {domB} has no supporting "
                f"cross-domain ACL/membership edges"))
        else:
            print(f"  ✓ {domA} ↔ {domB}: {count} supporting cross-domain edges")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hetero", required=True,
                        help="Path to heterodata.pt")
    parser.add_argument("--examples", default=None,
                        help="Path to training_examples.json (training mode)")
    parser.add_argument("--graph", default=None,
                        help="Path to forest_graph.json (for name resolution)")
    args = parser.parse_args()

    print("=" * 60)
    print("validate_dataset.py")
    print("=" * 60)

    warnings = []

    # load heterodata
    hetero_path = Path(args.hetero)
    print(f"Loading: {hetero_path}")
    try:
        data = torch.load(hetero_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"FATAL: cannot load heterodata: {e}")
        return 1

    # load forest for name resolution
    idx_to_name = {}
    if args.graph:
        graph_path = Path(args.graph)
        if graph_path.exists():
            print(f"Loading: {graph_path}")
            idx_to_name = load_idx_to_name(graph_path)

    # run checks
    check_node_features(data, warnings)
    check_edges(data, warnings)
    check_connectivity(data, idx_to_name, warnings)
    check_domains(data, idx_to_name, warnings)
    # build the edge_tensors dict from HeteroData first
    edge_tensors = {et[1]: data[et].edge_index for et in data.edge_types}
    check_cross_domain_coverage(data, edge_tensors, idx_to_name, warnings)

    # training examples if provided
    if args.examples:
        examples_path = Path(args.examples)
        if examples_path.exists():
            print(f"\nLoading: {examples_path}")
            examples = json.loads(examples_path.read_text(encoding="utf-8"))
            check_training_examples(examples, data["node"].num_nodes, warnings)
        else:
            print(f"\n  (examples file not found: {examples_path})")

    # final verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    fatal_count = sum(1 for f, _ in warnings if f)
    warn_count  = sum(1 for f, _ in warnings if not f)

    if fatal_count > 0:
        print(f"  ✗ {fatal_count} fatal issue(s) — do not proceed to training")
        for f, msg in warnings:
            if f:
                print(f"    {msg}")
        return 1

    if warn_count > 0:
        print(f"  ⚠ {warn_count} warning(s) — review before training:")
        for f, msg in warnings:
            if not f:
                print(f"    - {msg}")
        print(f"\n  Warnings are informational. Training can proceed.")
        return 0

    print("  ✓ All checks passed. Ready for training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())