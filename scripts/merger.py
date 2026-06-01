"""
merger.py
---------
Generic forest merger for the GNN-AD-Navigator pipeline.

Accepts heterogeneous inputs in any combination:
    - directory of cleaned bloodhound JSON files (e.g. *_users.json)
    - a single forest_graph.json file (e.g. already-stitched, prepared domain)

Outputs one merged forest_graph.json plus a summary of trust edges discovered.

Pipeline position:
    raw scan → filter → (stitch ADCS) → prepared.json
    ↓
    merger.py {prepared_a.json prepared_b.json ...} → forest_graph.json
    ↓
    build_dataset.py → heterodata.pt
    ↓
    train.py / infer.py

Usage:
    # mixed inputs
    python merger.py \\
        --input inlanefreight/prepared.json \\
                logistics/cleaned_big/ \\
                freightlogistics/cleaned_big/ \\
        --output testing_data/forest_graph.json

    # auto-discovery
    python merger.py \\
        --auto-input testing_data \\
        --output testing_data/forest_graph.json
"""

import json
import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter

NODE_TYPES = ["users", "computers", "groups", "domains",
              "gpos", "ous", "containers", "cas", "certtemplates"]


# ── normalisation ─────────────────────────────────────────────────────────────

def normalize_object(obj: dict) -> None:
    """Normalise SIDs (uppercase) and key string properties (lowercase) in place."""
    oid = obj.get("ObjectIdentifier", "")
    if isinstance(oid, str):
        obj["ObjectIdentifier"] = oid.upper()

    for ace in obj.get("Aces", []) or []:
        sid = ace.get("PrincipalSID", "")
        if isinstance(sid, str):
            ace["PrincipalSID"] = sid.upper()
        if "RightName" in ace and isinstance(ace["RightName"], str):
            ace["RightName"] = ace["RightName"].lower()

    props = obj.get("Properties", {})
    if isinstance(props, dict):
        for k in ["name", "domain", "distinguishedname", "samaccountname"]:
            v = props.get(k)
            if isinstance(v, str):
                props[k] = v.lower()

    # normalise trust targets to uppercase SIDs
    for trust in obj.get("Trusts", []) or []:
        tsid = trust.get("TargetDomainSid", "")
        if isinstance(tsid, str):
            trust["TargetDomainSid"] = tsid.upper()


# ── input handlers ───────────────────────────────────────────────────────────

def is_forest_graph_json(path: Path) -> bool:
    """Detect single-file forest_graph.json by looking at top-level keys."""
    try:
        head = json.loads(path.read_text(encoding="utf-8"))
        return any(k in head for k in NODE_TYPES)
    except Exception:
        return False


def load_forest_graph_file(path: Path) -> dict:
    """Load a single forest_graph.json and return its dict of {node_type: [objects]}."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    collection = {}
    for nt in NODE_TYPES:
        objs = raw.get(nt, [])
        for obj in objs:
            normalize_object(obj)
        collection[nt] = objs
    return collection


def discover_files_in_folder(folder: Path) -> dict:
    """Find *_<nodetype>.json files in a folder."""
    by_type = defaultdict(list)
    for f in sorted(folder.glob("*.json")):
        stem = f.stem.lower()
        for nt in NODE_TYPES:
            if stem.endswith(f"_{nt}") or stem == nt:
                by_type[nt].append(f)
                break
    return dict(by_type)


def load_folder(folder: Path) -> dict:
    """Load all per-type JSON files in a folder into a unified collection."""
    by_type = discover_files_in_folder(folder)
    collection = {}
    for nt, files in by_type.items():
        objects = []
        for f in files:
            try:
                raw  = json.loads(f.read_text(encoding="utf-8"))
                objs = raw.get("data", []) if isinstance(raw, dict) else []
                for obj in objs:
                    normalize_object(obj)
                objects.extend(objs)
            except Exception as e:
                print(f"    WARN: failed to load {f}: {e}", file=sys.stderr)
        collection[nt] = objects
    return collection


def load_input(path: Path) -> tuple[str, dict]:
    """
    Dispatch: returns (input_type, collection_dict).
    input_type is 'file' or 'folder' for the report.
    """
    if path.is_file() and is_forest_graph_json(path):
        return ("file", load_forest_graph_file(path))
    if path.is_dir():
        return ("folder", load_folder(path))
    raise ValueError(f"Unrecognised input: {path}  "
                      f"(must be a folder or a forest_graph.json file)")


# ── merge logic ──────────────────────────────────────────────────────────────

def score_obj(obj: dict) -> int:
    """Score node richness — used to pick the better record when deduplicating."""
    score = 0
    for v in obj.get("Properties", {}).values():
        if v is not None and v != "" and v is not False:
            score += 1
    score += len(obj.get("Aces", []) or [])
    score += len(obj.get("Members", []) or [])
    score += len(obj.get("Trusts", []) or []) * 5   # trusts are valuable
    return score


def merge(collections: list) -> dict:
    """Merge collections by ObjectIdentifier, keeping the richer record."""
    forest = {nt: {} for nt in NODE_TYPES}

    for coll in collections:
        for nt, objects in coll.items():
            if nt not in forest:
                forest[nt] = {}
            for obj in objects:
                oid = obj.get("ObjectIdentifier", "")
                if not oid:
                    continue
                if oid not in forest[nt] or score_obj(obj) > score_obj(forest[nt][oid]):
                    forest[nt][oid] = obj

    return {nt: list(d.values()) for nt, d in forest.items()}


# ── validation reports ──────────────────────────────────────────────────────

def report_node_counts(forest: dict) -> int:
    total = 0
    for nt in NODE_TYPES:
        n = len(forest.get(nt, []))
        if n > 0:
            print(f"  {nt:<15}: {n}")
        total += n
    print(f"  {'TOTAL':<15}: {total}")
    return total


def report_trusts(forest: dict) -> int:
    """Count and print all trusts discovered in domain objects."""
    print("\nTrusts inventory:")
    trust_total = 0
    for dom in forest.get("domains", []):
        name   = dom.get("Properties", {}).get("name", "?")
        trusts = dom.get("Trusts", []) or []
        if not trusts:
            print(f"  {name}: no trusts captured")
            continue
        for t in trusts:
            target = t.get("TargetDomainName", "?")
            ttype  = t.get("TrustType", "?")
            tdir   = t.get("TrustDirection", "?")
            sidf   = t.get("SidFilteringEnabled", "?")
            print(f"  {name} → {target}  type={ttype} "
                  f"direction={tdir} sid_filter={sidf}")
            trust_total += 1
    print(f"  Total trust relationships: {trust_total}")
    return trust_total


def report_edges(forest: dict, retained_sids: set) -> dict:
    """Aggregate edge type counts and dangling-reference rate."""
    counts   = Counter()
    dangling = 0
    total    = 0

    for nt, objs in forest.items():
        for obj in objs:
            for ace in obj.get("Aces", []) or []:
                total += 1
                rel = (ace.get("RightName") or "unknown").lower()
                counts[rel] += 1
                if (ace.get("PrincipalSID") or "").upper() not in retained_sids:
                    dangling += 1
            for key in ["Members", "ChildObjects", "Links"]:
                for item in obj.get(key, []) or []:
                    total += 1
                    counts[key.lower()] += 1
                    tid = (item if isinstance(item, str)
                           else item.get("ObjectIdentifier", "")).upper()
                    if tid not in retained_sids:
                        dangling += 1

    print(f"\nEdge inventory ({len(counts)} types, "
          f"{total} total references, {dangling} dangling = "
          f"{100*dangling/max(total,1):.1f}%):")
    for rel, n in sorted(counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {rel:<35}: {n}")
    if len(counts) > 15:
        print(f"  ... and {len(counts)-15} more")
    return {"total": total, "dangling": dangling, "by_type": dict(counts)}


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generic pipeline-ready forest merger.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", nargs="+",
                     help="One or more inputs (folders or forest_graph.json files).")
    src.add_argument("--auto-input",
                     help="Parent folder — auto-discovers subfolders with node JSON.")
    parser.add_argument("--output", required=True,
                        help="Path to output forest_graph.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # ─ resolve inputs ─
    if args.input:
        input_paths = [Path(p) for p in args.input]
    else:
        parent = Path(args.auto_input)
        if not parent.is_dir():
            print(f"ERROR: {parent} is not a directory", file=sys.stderr)
            return 1
        input_paths = []
        # find both prepared.json files and folders with *_users.json
        for p in sorted(parent.rglob("*")):
            if p.is_file() and p.suffix == ".json" and is_forest_graph_json(p):
                input_paths.append(p)
            elif p.is_dir() and any(p.glob("*_users.json")):
                input_paths.append(p)
        if not input_paths:
            print(f"ERROR: no inputs found under {parent}", file=sys.stderr)
            return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("=" * 60)
        print("merger.py")
        print("=" * 60)
        print(f"Inputs : {len(input_paths)}")
        for p in input_paths:
            kind = "file" if p.is_file() else "folder"
            print(f"  [{kind}]  {p}")
        print(f"Output : {output_path}")
        print()

    # ─ load + merge ─
    collections = []
    for p in input_paths:
        if not args.quiet:
            print(f"Loading {p.name}...")
        try:
            kind, coll = load_input(p)
            n = sum(len(v) for v in coll.values())
            if not args.quiet:
                print(f"  [{kind}] {n} nodes")
            collections.append(coll)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 1

    if not args.quiet:
        print("\nMerging...")
    forest = merge(collections)

    # ─ validate ─
    retained = {(obj.get("ObjectIdentifier") or "").upper()
                for objs in forest.values()
                for obj in objs
                if obj.get("ObjectIdentifier")}

    if not args.quiet:
        print("\nMerged forest:")
        total_nodes = report_node_counts(forest)
        trust_total = report_trusts(forest)
        edge_stats  = report_edges(forest, retained)

    # ─ write output ─
    # also embed minimal merge metadata for traceability
    output = {nt: forest.get(nt, []) for nt in NODE_TYPES}
    output["_merge_metadata"] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs":       [str(p) for p in input_paths],
        "node_count":   sum(len(forest.get(nt, [])) for nt in NODE_TYPES),
        "trust_count":  trust_total if not args.quiet else None,
    }

    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if not args.quiet:
        kb = output_path.stat().st_size // 1024
        print(f"\nWritten {output_path}  ({kb} KB)")
        print("\nNext: build_dataset.py — converts forest_graph.json into PyG HeteroData")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())