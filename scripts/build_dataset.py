"""
build_dataset.py
----------------
Converts a merged forest_graph.json into true PyG HeteroData tensors.
"""

import json
import csv
import argparse
import torch
from pathlib import Path
from collections import defaultdict
from torch_geometric.data import HeteroData

# ── node feature specs ────────────────────────────────────────────────────────

NODE_FEATURES = {
    "users":      [("enabled",True),("admincount",False),("hasspn",False),
                    ("dontreqpreauth",False),("pwdneverexpires",False),
                    ("unconstraineddelegation",False)],
    "computers":  [("enabled",True),("unconstraineddelegation",False),
                    ("haslaps",False),("admincount",False)],
    "groups":     [("admincount",False)],
    "domains":    [("functionallevel",0)],
    "gpos":       [],
    "ous":        [],
    "containers": [],
    "cas":        [("web_enrollment",False),("esc7",False),("esc8",False)],
    "certtemplates": [("enabled",True),("client_authentication",False),
                       ("enrollee_supplies_subject",False),
                       ("requires_manager_approval",False),("any_purpose",False),
                       ("esc1",False),("esc2",False),("esc3",False),
                       ("esc4",False),("esc9",False),("esc13",False)],
}
NODE_ORDER = list(NODE_FEATURES.keys())
MAX_FEAT   = max(len(v) for v in NODE_FEATURES.values()) + 3  # +3 extras

TRUST_EDGE_ALIASES = {
    "sameforesttrust":     "sameforesttrust",
    "crossforesttrust":    "crossforesttrust",
    "trustedby":           "sameforesttrust",
    "spoofsidhistory":     "sameforesttrust",
    "abusetgtdelegation":  "sameforesttrust",
}

def trust_type_to_edge_name(trust_type: str) -> str:
    if trust_type in ("ParentChild", "TreeRoot"): return "sameforesttrust"
    if trust_type == "Forest": return "crossforesttrust"
    return "sameforesttrust"

def canonicalize_edge_name(rel: str) -> str:
    rel = rel.lower()
    return TRUST_EDGE_ALIASES.get(rel, rel)

# ── feature helpers ──────────────────────────────────────────────────────────

SERVICE_HINTS = ["svc","service","sql","iis","exchange","sccm",
                  "mssql","backup","krbtgt","scan","mgmt"]

def ou_depth(dn: str) -> float:
    return float((dn or "").upper().count(",OU=")) / 5.0

def is_service(props: dict) -> float:
    name = (props.get("name") or "").lower()
    return float(bool(props.get("hasspn", False))
                 or any(h in name for h in SERVICE_HINTS))

def extract_features(obj: dict, node_type: str) -> list:
    props = obj.get("Properties", {}) or {}
    spec  = NODE_FEATURES.get(node_type, [])
    feats = []
    
    for key, default in spec:
        v = props.get(key, default)
        if isinstance(v, bool):
            feats.append(float(v))
        elif isinstance(v, (int, float)):
            feats.append(min(float(v), 10.0) / 10.0)
        else:
            feats.append(0.0)
            
    if node_type in ("users", "computers"):
        feats.append(is_service(props))
        feats.append(ou_depth(props.get("distinguishedname", "")))
    else:
        feats.extend([0.0, 0.0])
        
    # BUGFIX: Pad FIRST, append objective LAST
    while len(feats) < MAX_FEAT - 1:
        feats.append(0.0)
    feats.append(0.0)  # is_objective guaranteed at index [-1]
    
    return feats[:MAX_FEAT]

# ── load graph and build indices ─────────────────────────────────────────────

def load_graph(path: Path):
    forest = json.loads(path.read_text(encoding="utf-8"))

    global_map  = {}   # SID -> (node_type, local_idx)
    name_to_sid = {}
    sid_to_global = {} # Keep global ID mapping for legacy compatibility
    feat_rows_dict = {nt: [] for nt in NODE_ORDER}
    
    global_counter = 0

    for nt in NODE_ORDER:
        for obj in forest.get(nt, []) or []:
            oid  = (obj.get("ObjectIdentifier") or "").upper()
            if not oid or oid in global_map:
                continue

            name = (obj.get("Properties", {}).get("name") or "").lower()
            local_idx = len(feat_rows_dict[nt])
            
            global_map[oid] = (nt, local_idx)
            sid_to_global[oid] = global_counter
            name_to_sid[name] = oid

            short = name.split("@")[0].split(".")[0]
            if short and short not in name_to_sid:
                name_to_sid[short] = oid

            feat_rows_dict[nt].append(extract_features(obj, nt))
            global_counter += 1

    # Null credential sentinel for Unconstrained mode
    sentinel_local_idx = len(feat_rows_dict["users"])
    feat_rows_dict["users"].append([0.0] * MAX_FEAT)
    sentinel_global_idx = global_counter

    X_dict = {nt: torch.tensor(rows, dtype=torch.float32) for nt, rows in feat_rows_dict.items() if rows}
    
    return forest, global_map, name_to_sid, sid_to_global, X_dict, sentinel_local_idx, sentinel_global_idx

# ── edge construction (Hetero Triplet Upgrade) ───────────────────────────────

def build_edges(forest: dict, global_map: dict) -> dict:
    edges = defaultdict(list)

    for nt in NODE_ORDER:
        for obj in forest.get(nt, []) or []:
            target_sid = (obj.get("ObjectIdentifier") or "").upper()
            if target_sid not in global_map:
                continue
            t_type, t_local = global_map[target_sid]

            for ace in obj.get("Aces", []) or []:
                p_sid = (ace.get("PrincipalSID") or "").upper()
                if p_sid in global_map:
                    p_type, p_local = global_map[p_sid]
                    rel = canonicalize_edge_name(ace.get("RightName") or "unknown")
                    edges[(p_type, rel, t_type)].append((p_local, t_local))

            for member in obj.get("Members", []) or []:
                m_sid = (member if isinstance(member, str) else member.get("ObjectIdentifier", "")).upper()
                if m_sid in global_map:
                    m_type, m_local = global_map[m_sid]
                    edges[(m_type, "memberof", t_type)].append((m_local, t_local))

            for child in obj.get("ChildObjects", []) or []:
                c_sid = (child if isinstance(child, str) else child.get("ObjectIdentifier", "")).upper()
                if c_sid in global_map:
                    c_type, c_local = global_map[c_sid]
                    edges[(t_type, "contains", c_type)].append((t_local, c_local))

            for link in obj.get("Links", []) or []:
                l_sid = (link if isinstance(link, str) else (link.get("GUID") or link.get("ObjectIdentifier") or "")).upper()
                if l_sid in global_map:
                    l_type, l_local = global_map[l_sid]
                    edges[(l_type, "gplink", t_type)].append((l_local, t_local))

            for d in obj.get("AllowedToDelegate", []) or []:
                d_sid = (d if isinstance(d, str) else d.get("ObjectIdentifier", "")).upper()
                if d_sid in global_map:
                    d_type, d_local = global_map[d_sid]
                    edges[(t_type, "allowedtodelegate", d_type)].append((t_local, d_local))

            for a in obj.get("AllowedToAct", []) or []:
                a_sid = (a if isinstance(a, str) else a.get("ObjectIdentifier", "")).upper()
                if a_sid in global_map:
                    a_type, a_local = global_map[a_sid]
                    edges[(a_type, "allowedtoact", t_type)].append((a_local, t_local))

            for h in obj.get("HasSIDHistory", []) or []:
                h_sid = (h if isinstance(h, str) else h.get("ObjectIdentifier", "")).upper()
                if h_sid in global_map:
                    h_type, h_local = global_map[h_sid]
                    edges[(t_type, "hassidhistory", h_type)].append((t_local, h_local))

            # BUGFIX: HasSession edge capture
            for s in obj.get("Sessions", []) or []:
                s_sid = (s if isinstance(s, str) else s.get("ObjectIdentifier", "")).upper()
                if s_sid in global_map:
                    s_type, s_local = global_map[s_sid]
                    edges[(t_type, "hassession", s_type)].append((t_local, s_local))

            if nt == "domains":
                for trust in obj.get("Trusts", []) or []:
                    td_sid = (trust.get("TargetDomainSid") or "").upper()
                    if td_sid in global_map:
                        td_type, td_local = global_map[td_sid]
                        rel = trust_type_to_edge_name(trust.get("TrustType", ""))
                        edges[(t_type, rel, td_type)].append((t_local, td_local))
                        if trust.get("TrustDirection") == "Bidirectional":
                            edges[(td_type, rel, t_type)].append((td_local, t_local))

    edge_tensors = {}
    for triplet, pairs in edges.items():
        if pairs:
            # Deduplicate and format for PyG
            unique_pairs = list(set(pairs))
            edge_tensors[triplet] = torch.tensor(unique_pairs, dtype=torch.long).t().contiguous()

    return edge_tensors

def build_heterodata(X_dict: dict, edge_tensors: dict) -> HeteroData:
    data = HeteroData()
    for nt, X in X_dict.items():
        data[nt].x = X
        data[nt].num_nodes = X.shape[0]
    for triplet, ei in edge_tensors.items():
        data[triplet].edge_index = ei
    return data

# ── zoom.csv handling for training mode ──────────────────────────────────────

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

def load_zoom(path: Path, name_to_sid: dict, global_map: dict, sid_to_global: dict,
              X_dict: dict, sentinel_local: int, sentinel_global: int) -> tuple[list, set]:
    
    paths = defaultdict(list)
    raw_rows = []

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_rows.append(row)
            paths[row["path_id"].strip()].append(row)

    objective_sids = set()
    for pid, rows in paths.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["step"]))
        last_hop_sid = resolve_sid(rows_sorted[-1]["next_hop_id"], name_to_sid, global_map)
        if last_hop_sid:
            objective_sids.add(last_hop_sid)
            nt, local_idx = global_map[last_hop_sid]
            X_dict[nt][local_idx, -1] = 1.0   # Set objective locally

    examples = []
    skipped  = 0

    for row in raw_rows:
        path_id    = row["path_id"].strip()
        step       = int(row["step"])
        train_on   = int(row.get("train_on_step", "1") or "1")
        eff_weight = float(row.get("path_weight", "1.0") or "1.0") * (1.0 - float(row.get("step_cost", "0.0") or "0.0"))

        if train_on == 0:
            continue

        cur_sid  = resolve_sid(row["current_node_id"], name_to_sid, global_map)
        hop_sid  = resolve_sid(row["next_hop_id"], name_to_sid, global_map)
        cred_sid = resolve_sid(row["credential_used_id"], name_to_sid, global_map)

        if not cur_sid or not hop_sid or not cred_sid:
            print(f"  WARN [{path_id} step {step}]: unresolved nodes — skipped")
            skipped += 1
            continue

        cur_type, cur_local = global_map[cur_sid]
        hop_type, hop_local = global_map[hop_sid]

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
                # Typed coordinates (For Heterogeneous Models)
                "current_type":     cur_type,
                "current_local":    cur_local,
                "credential_type":  cred_type,
                "credential_local": cred_local,
                "next_hop_type":    hop_type,
                "next_hop_local":   hop_local,
                # Global coordinates (For backwards compatibility)
                "current_idx":      sid_to_global[cur_sid],
                "credential_idx":   cred_global,
                "next_hop_idx":     sid_to_global[hop_sid],
                "eff_weight":       round(eff_weight, 4),
                "notes":            row.get("notes", ""),
            })

    return examples, objective_sids, skipped

# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="Path to merged forest_graph.json")
    parser.add_argument("--zoom",  default=None, help="Optional zoom.csv for training mode")
    parser.add_argument("--out",   required=True, help="Output directory")
    args = parser.parse_args()

    graph_path, out_dir = Path(args.graph), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("build_dataset.py (HGT Upgrade)")
    print("=" * 60)

    print("[1] Loading heterogeneous graph...")
    forest, global_map, name_to_sid, sid_to_global, X_dict, sentinel_local, sentinel_global = load_graph(graph_path)
    print(f"  Nodes      : {len(global_map)}")
    
    print("\n[2] Building heterogeneous edges...")
    edge_tensors = build_edges(forest, global_map)
    print(f"  Edge Triplet Types : {len(edge_tensors)}")

    objectives = set()
    examples   = []
    if args.zoom:
        print(f"\n[3] Parsing {args.zoom}...")
        examples, objectives, skipped = load_zoom(
            Path(args.zoom), name_to_sid, global_map, sid_to_global, X_dict, sentinel_local, sentinel_global
        )
        print(f"  Examples   : {len(examples)}")
        print(f"  Objectives : {len(objectives)}")

    print("\n[4] Assembling HeteroData...")
    data = build_heterodata(X_dict, edge_tensors)
    for nt in data.node_types:
        print(f"  data['{nt}'].x : {data[nt].x.shape}")
    print(f"  Total Edge Types: {len(data.edge_types)}")

    print("\n[5] Saving...")
    torch.save(data, out_dir / "heterodata.pt")
    if args.zoom:
        (out_dir / "training_examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)

if __name__ == "__main__":
    main()