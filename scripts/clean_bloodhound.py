#!/usr/bin/env python3
"""
clean_bloodhound.py
Cleans and filters raw BloodHound CE JSON files for GNN ingestion.
Input:  directory of extracted BloodHound JSON files
Output: cleaned JSON files + stats report
"""

import json
import os
import argparse
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
MIN_EDGE_DEGREE   = 1   # drop nodes with zero edges after filtering
KEEP_DISABLED_IF_HAS_ACL = True

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(data, path):
    # Preserve CE format on output
    if "meta" in data:
        out = {
            "data": data["data"] if "data" in data else [],
            "meta": data["meta"]
        }
        # Update count to reflect cleaning
        key = data["meta"].get("type", "")
        if key in data.get("data", {}):
            out["meta"]["count"] = len(out["data"])
    else:
        out = data
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {path}")

def build_edge_index(all_data):
    edge_index = defaultdict(set)
    edge_types  = defaultdict(list)

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

                # CE format wraps session/localadmin in {"Collected":, "Results":}
                if isinstance(raw, dict):
                    raw = raw.get("Results", [])

                for rel in raw:
                    # Skip if not a dict (some blocks contain plain strings)
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

    # Never drop if has meaningful ACL edges
    dangerous_rights = {"GenericAll","WriteDacl","WriteOwner",
                        "GenericWrite","AddKeyCredentialLink",
                        "AllExtendedRights","ForceChangePassword"}
    has_dangerous_ace = any(
        a.get("RightName") in dangerous_rights for a in aces
    )
    if has_dangerous_ace:
        return False

    # Never drop DCs
    if props.get("unconstraineddelegation", False):
        return False

    # Drop if zero edges and matches decoration pattern
    decoration_patterns = ["LON-07","LON-06","LON-05","LON-04",
                           "LON-03","WKS-0","LPTP-0"]
    name = props.get("name","").upper()
    is_pattern = any(p in name for p in decoration_patterns)
    has_edges  = len(edge_index.get(sid, set())) > MIN_EDGE_DEGREE

    return is_pattern and not has_edges

def should_drop_user(obj, edge_index):
    """
    Drop users that are noise:
    - Disabled AND no ACL edges AND only in Domain Users
    """
    sid   = obj.get("ObjectIdentifier", "")
    props = obj.get("Properties", {})

    enabled = props.get("enabled", True)
    aces    = obj.get("Aces", [])
    has_acl_edges = len(aces) > 0

    # Never drop if has outgoing ACL edges (dangerous even if disabled)
    if KEEP_DISABLED_IF_HAS_ACL and has_acl_edges:
        return False

    # Never drop if has sessions or admin rights
    has_sessions = len(obj.get("Sessions", {}).get("Results", [])) > 0
    if has_sessions:
        return False

    # Drop if disabled and no meaningful edges
    if not enabled and not has_acl_edges:
        neighbors = edge_index.get(sid, set())
        if len(neighbors) <= 1:  # only domain membership
            return True

    return False

def should_drop_group(obj, edge_index):
    """Drop empty groups with no ACL context."""
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

    # Load all JSON files
    all_data = {}
    file_map = {}
    for fname in os.listdir(input_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(input_dir, fname)
        print(f"Loading: {path}")
        data = load_json(path)
        
        for key in ["computers","users","groups","domains",
                    "gpos","ous","containers"]:
            # BloodHound CE format
            if "data" in data and "meta" in data:
                meta_type = data["meta"].get("type", "").lower()
                if meta_type == key:
                    all_data[key] = data["data"]
                    file_map[key] = (fname, data)
                    break
            # Legacy format fallback
            elif key in data:
                all_data[key] = data[key]
                file_map[key] = (fname, data)
                break

    print(f"\n{'='*50}")
    print(f"Loaded types: {list(all_data.keys())}")

    # Build edge index across everything
    print("Building edge index...")
    edge_index, edge_types = build_edge_index(all_data)

    stats = defaultdict(lambda: {"before": 0, "after": 0, "dropped": 0})

    # ── Filter Computers ──
    if "computers" in all_data:
        before = len(all_data["computers"])
        cleaned = [c for c in all_data["computers"]
                   if not is_decoration_computer(c, edge_index)]
        after = len(cleaned)
        all_data["computers"] = cleaned
        stats["computers"] = {"before": before, "after": after,
                               "dropped": before - after}

    # ── Filter Users ──
    if "users" in all_data:
        before = len(all_data["users"])
        cleaned = [u for u in all_data["users"]
                   if not should_drop_user(u, edge_index)]
        after = len(cleaned)
        all_data["users"] = cleaned
        stats["users"] = {"before": before, "after": after,
                           "dropped": before - after}

    # ── Filter Groups ──
    if "groups" in all_data:
        before = len(all_data["groups"])
        cleaned = [g for g in all_data["groups"]
                   if not should_drop_group(g, edge_index)]
        after = len(cleaned)
        all_data["groups"] = cleaned
        stats["groups"] = {"before": before, "after": after,
                            "dropped": before - after}

    # ── Keep domains/gpos/ous/containers as-is ──
    for key in ["domains", "gpos", "ous", "containers"]:
        if key in all_data:
            stats[key] = {"before": len(all_data[key]),
                          "after":  len(all_data[key]),
                          "dropped": 0}

    # ── Save cleaned files ──
    print(f"\n{'='*50}")
    print("Saving cleaned files...")
    for key, (fname, original_data) in file_map.items():
        if key in all_data:
            out = dict(original_data)
            out[key] = all_data[key]
            save_json(out, os.path.join(output_dir, f"cleaned_{fname}"))

    # ── Stats report ──
    print(f"\n{'='*50}")
    print(f"{'Type':<15} {'Before':>8} {'After':>8} {'Dropped':>8}")
    print("-" * 45)
    total_before = total_after = 0
    for t, s in stats.items():
        print(f"{t:<15} {s['before']:>8} {s['after']:>8} {s['dropped']:>8}")
        total_before += s["before"]
        total_after  += s["after"]
    print("-" * 45)
    print(f"{'TOTAL':<15} {total_before:>8} {total_after:>8} "
          f"{total_before-total_after:>8}")
    print(f"\nEdge index covers {len(edge_index)} unique SIDs")
    print(f"Unique edge type pairs: {len(edge_types)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,
                        help="Directory with extracted BloodHound JSONs")
    parser.add_argument("--output", required=True,
                        help="Output directory for cleaned JSONs")
    args = parser.parse_args()
    clean(args.input, args.output)