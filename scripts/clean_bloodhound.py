#!/usr/bin/env python3
"""
clean_bloodhound.py
Cleans and filters raw BloodHound CE JSON files for GNN ingestion.

Input:  directory (or directory tree) of BloodHound CE JSON files
Output: cleaned, merged JSON files + stats report

This version accumulates across multiple files per node type — so when
the input contains scans from several domains (or a single domain split
across multiple files), no data is silently overwritten. The directory
is also walked recursively, so subfolders are picked up automatically.
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
MIN_EDGE_DEGREE   = 1
KEEP_DISABLED_IF_HAS_ACL = True

NODE_TYPES = ["computers", "users", "groups", "domains",
              "gpos", "ous", "containers"]


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def discover_input_files(input_dir):
    """
    Walks the input directory (and subdirectories) and returns a dict mapping
    node_type → list of records accumulated from all matching files.
    Also returns a dict mapping node_type → representative metadata block
    (used to preserve CE format on save).
    """
    input_path = Path(input_dir)
    accumulated = {nt: [] for nt in NODE_TYPES}
    meta_template = {}
    source_files = defaultdict(list)

    json_files = sorted(input_path.rglob("*.json"))
    if not json_files:
        print(f"WARNING: no JSON files found under {input_dir}")
        return accumulated, meta_template, source_files

    for json_path in json_files:
        rel = json_path.relative_to(input_path)
        try:
            data = load_json(json_path)
        except Exception as e:
            print(f"  Skipping {rel} (parse error: {e})")
            continue

        # detect node type from CE metadata first
        node_type = None
        if isinstance(data, dict) and "meta" in data:
            meta_type = (data["meta"].get("type") or "").lower()
            if meta_type in NODE_TYPES:
                node_type = meta_type

        # fall back to legacy format (top-level node-type keys)
        if node_type is None and isinstance(data, dict):
            for k in NODE_TYPES:
                if k in data and isinstance(data[k], list):
                    node_type = k
                    break

        if node_type is None:
            # not a recognised bloodhound file (could be certipy, etc.)
            print(f"  Ignoring {rel} (not a recognised bloodhound scan)")
            continue

        # extract the records — handle both CE and legacy shapes
        if "data" in data:
            records = data["data"] or []
        else:
            records = data.get(node_type, []) or []

        accumulated[node_type].extend(records)
        source_files[node_type].append(str(rel))

        # capture the first meta we see for this type (used on save)
        if node_type not in meta_template and "meta" in data:
            meta_template[node_type] = data["meta"]

        print(f"  Loaded {rel}: {len(records)} {node_type}")

    return accumulated, meta_template, source_files


def build_edge_index(all_data):
    edge_index = defaultdict(set)
    edge_types = defaultdict(list)

    for dtype, items in all_data.items():
        for obj in items:
            sid = obj.get("ObjectIdentifier", "")
            if not sid:
                continue
            for rel_block in ["Aces", "Members", "Sessions",
                               "LocalAdmins", "RemoteDesktopUsers",
                               "DcomUsers", "PSRemoteUsers",
                               "AllowedToDelegate", "AllowedToAct",
                               "HasSIDHistory"]:
                raw = obj.get(rel_block, [])
                if isinstance(raw, dict):
                    raw = raw.get("Results", [])
                for rel in raw:
                    if not isinstance(rel, dict):
                        continue
                    target = rel.get("ObjectIdentifier", "") or \
                              rel.get("MemberId", "")
                    if target:
                        edge_index[sid].add(target)
                        edge_index[target].add(sid)
                        edge_types[(sid, target)].append(rel_block)

    return edge_index, edge_types


def is_decoration_computer(obj, edge_index):
    sid   = obj.get("ObjectIdentifier", "")
    props = obj.get("Properties", {})
    aces  = obj.get("Aces", [])
    dangerous_rights = {"GenericAll","WriteDacl","WriteOwner",
                         "GenericWrite","AddKeyCredentialLink",
                         "AllExtendedRights","ForceChangePassword"}
    if any(a.get("RightName") in dangerous_rights for a in aces):
        return False
    if props.get("unconstraineddelegation", False):
        return False
    decoration_patterns = ["LON-07","LON-06","LON-05","LON-04",
                            "LON-03","WKS-0","LPTP-0"]
    name = props.get("name","").upper()
    is_pattern = any(p in name for p in decoration_patterns)
    has_edges  = len(edge_index.get(sid, set())) > MIN_EDGE_DEGREE
    return is_pattern and not has_edges


def should_drop_user(obj, edge_index):
    sid   = obj.get("ObjectIdentifier", "")
    props = obj.get("Properties", {})
    enabled = props.get("enabled", True)
    aces    = obj.get("Aces", [])
    has_acl_edges = len(aces) > 0
    if KEEP_DISABLED_IF_HAS_ACL and has_acl_edges:
        return False
    has_sessions = len(obj.get("Sessions", {}).get("Results", [])) > 0
    if has_sessions:
        return False
    if not enabled and not has_acl_edges:
        neighbors = edge_index.get(sid, set())
        if len(neighbors) <= 1:
            return True
    return False


def should_drop_group(obj, edge_index):
    sid     = obj.get("ObjectIdentifier", "")
    members = obj.get("Members", [])
    aces    = obj.get("Aces", [])
    if len(members) == 0 and len(aces) == 0:
        if len(edge_index.get(sid, set())) == 0:
            return True
    return False


# ── Main ─────────────────────────────────────────────────────────────────────
def clean(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning input tree: {input_dir}")
    print("=" * 60)
    all_data, meta_template, source_files = discover_input_files(input_dir)

    # report what we accumulated
    print("\n" + "=" * 60)
    print("Accumulated raw records:")
    for nt in NODE_TYPES:
        n = len(all_data.get(nt, []))
        sources = len(source_files.get(nt, []))
        if n > 0:
            print(f"  {nt:<12}: {n} records from {sources} file(s)")

    # deduplicate by ObjectIdentifier in case the same domain was scanned twice
    print("\nDeduplicating by ObjectIdentifier...")
    for nt in NODE_TYPES:
        if nt not in all_data: continue
        seen = {}
        for obj in all_data[nt]:
            oid = obj.get("ObjectIdentifier", "")
            if not oid:
                continue
            # keep the richer record if duplicate
            if oid not in seen or len(json.dumps(obj)) > len(json.dumps(seen[oid])):
                seen[oid] = obj
        before = len(all_data[nt])
        all_data[nt] = list(seen.values())
        after = len(all_data[nt])
        if before != after:
            print(f"  {nt}: {before} → {after} (removed {before-after} duplicates)")

    # build edge index across everything
    print("\nBuilding edge index...")
    edge_index, edge_types = build_edge_index(all_data)
    print(f"  Edge index covers {len(edge_index)} unique SIDs")

    stats = defaultdict(lambda: {"before": 0, "after": 0, "dropped": 0})

    # ── Filter Computers ──
    if all_data.get("computers"):
        before = len(all_data["computers"])
        cleaned = [c for c in all_data["computers"]
                    if not is_decoration_computer(c, edge_index)]
        all_data["computers"] = cleaned
        stats["computers"] = {"before": before, "after": len(cleaned),
                               "dropped": before - len(cleaned)}

    # ── Filter Users ──
    if all_data.get("users"):
        before = len(all_data["users"])
        cleaned = [u for u in all_data["users"]
                    if not should_drop_user(u, edge_index)]
        all_data["users"] = cleaned
        stats["users"] = {"before": before, "after": len(cleaned),
                           "dropped": before - len(cleaned)}

    # ── Filter Groups ──
    if all_data.get("groups"):
        before = len(all_data["groups"])
        cleaned = [g for g in all_data["groups"]
                    if not should_drop_group(g, edge_index)]
        all_data["groups"] = cleaned
        stats["groups"] = {"before": before, "after": len(cleaned),
                            "dropped": before - len(cleaned)}

    # ── Keep domains/gpos/ous/containers as-is ──
    for key in ["domains", "gpos", "ous", "containers"]:
        if all_data.get(key):
            stats[key] = {"before": len(all_data[key]),
                          "after":  len(all_data[key]),
                          "dropped": 0}

    # ── Save cleaned files — one per node type, accumulated ──
    print(f"\n{'='*60}")
    print("Saving cleaned files...")
    for nt in NODE_TYPES:
        records = all_data.get(nt, [])
        if not records:
            continue
        meta = meta_template.get(nt, {"type": nt})
        meta["count"] = len(records)
        out = {"data": records, "meta": meta}
        out_path = os.path.join(output_dir, f"cleaned_{nt}.json")
        save_json(out, out_path)

    # ── Stats report ──
    print(f"\n{'='*60}")
    print(f"{'Type':<15} {'Before':>8} {'After':>8} {'Dropped':>8}")
    print("-" * 45)
    total_before = total_after = 0
    for t in NODE_TYPES:
        if t in stats:
            s = stats[t]
            print(f"{t:<15} {s['before']:>8} {s['after']:>8} {s['dropped']:>8}")
            total_before += s["before"]
            total_after  += s["after"]
    print("-" * 45)
    print(f"{'TOTAL':<15} {total_before:>8} {total_after:>8} "
          f"{total_before-total_after:>8}")
    print(f"\nEdge index covers {len(edge_index)} unique SIDs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,
                        help="Directory (or directory tree) of BloodHound JSONs")
    parser.add_argument("--output", required=True,
                        help="Output directory for cleaned JSONs")
    args = parser.parse_args()
    clean(args.input, args.output)