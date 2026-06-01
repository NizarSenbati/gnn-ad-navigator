"""
build_dataset.py
----------------
Converts a merged forest_graph.json into the tensors that the model
consumes: a PyG HeteroData object plus a JSON list of training examples.

Key responsibilities of this stage:
    1. Translate BloodHound's nested JSON into edge_index tensors.
    2. Apply BloodHound's standard direction convention:
           principal → target  (the principal HAS the right ON the target)
    3. Extract trust edges from the Trusts array on domain objects.
    4. Canonicalise edge type names (lowercase, alias variants).
    5. Build state vectors and labels from zoom.csv (training mode only).

INPUTS
    results/forest_graph.json     — merged + stitched graph (from merger.py)
    results/zoom.csv              — expert traces for training (optional)

OUTPUTS
    results/heterodata.pt         — torch_geometric HeteroData (graph + features)
    results/training_examples.json — list of training example dicts (if zoom.csv present)

Run modes:
    Training data preparation:
        python build_dataset.py --graph results/forest_graph.json \\
                                 --zoom  results/zoom.csv \\
                                 --out   results/

    Inference data preparation (no zoom.csv):
        python build_dataset.py --graph testing_data/forest_graph.json \\
                                 --out   testing_data/
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

# ── edge type canonicalization ────────────────────────────────────────────────

# all trust-like edges normalized to one of these canonical names
TRUST_EDGE_ALIASES = {
    "sameforesttrust":     "sameforesttrust",
    "crossforesttrust":    "crossforesttrust",
    "trustedby":           "sameforesttrust",
    "spoofsidhistory":     "sameforesttrust",
    "abusetgtdelegation":  "sameforesttrust",
}

def trust_type_to_edge_name(trust_type: str) -> str:
    """Map BloodHound's TrustType field to a canonical edge name."""
    if trust_type in ("ParentChild", "TreeRoot"):
        return "sameforesttrust"
    if trust_type == "Forest":
        return "crossforesttrust"
    return "sameforesttrust"

def canonicalize_edge_name(rel: str) -> str:
    """Normalize edge type name. Lowercased + trust aliases applied."""
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
    feats.append(0.0)  # is_objective — set later from zoom.csv
    while len(feats) < MAX_FEAT:
        feats.append(0.0)
    return feats[:MAX_FEAT]

# ── load graph and build indices ─────────────────────────────────────────────

def load_graph(path: Path):
    forest = json.loads(path.read_text(encoding="utf-8"))

    id_to_idx  = {}
    name_to_id = {}
    feat_rows  = []

    for nt in NODE_ORDER:
        for obj in forest.get(nt, []) or []:
            oid  = (obj.get("ObjectIdentifier") or "").upper()
            if not oid or oid in id_to_idx:
                continue

            name = (obj.get("Properties", {}).get("name") or "").lower()
            idx  = len(id_to_idx)
            id_to_idx[oid]   = idx
            name_to_id[name] = oid

            short = name.split("@")[0].split(".")[0]
            if short and short not in name_to_id:
                name_to_id[short] = oid

            feat_rows.append(extract_features(obj, nt))

    # null credential sentinel — used for Mode 2 (unconstrained) inference
    feat_rows.append([0.0] * MAX_FEAT)

    X = torch.tensor(feat_rows, dtype=torch.float32)
    return forest, id_to_idx, name_to_id, X

# ── edge construction (the core fix) ─────────────────────────────────────────

def build_edges(forest: dict, id_to_idx: dict) -> dict:
    """
    Edge direction convention applied here:

        ACL edges      : principal → target object   (BloodHound standard)
        MemberOf       : member → group              (BloodHound standard)
        Trust edges    : source_domain → target_domain
        ChildObjects   : parent → child container

    These all match what BloodHound GUI renders.
    """
    known = set(id_to_idx.keys())
    edges = defaultdict(list)

    for nt in NODE_ORDER:
        for obj in forest.get(nt, []) or []:
            target_sid = (obj.get("ObjectIdentifier") or "").upper()
            if target_sid not in known:
                continue
            target_idx = id_to_idx[target_sid]

            # ── ACE-based edges: principal → target ──
            for ace in obj.get("Aces", []) or []:
                principal_sid = (ace.get("PrincipalSID") or "").upper()
                if principal_sid not in known:
                    continue
                rel = canonicalize_edge_name(ace.get("RightName") or "unknown")
                edges[rel].append((id_to_idx[principal_sid], target_idx))

            # ── Members: member → group ──
            # 'obj' here is the group; entries in Members are the members
            for member in obj.get("Members", []) or []:
                m_sid = (member if isinstance(member, str)
                         else member.get("ObjectIdentifier", "")).upper()
                if m_sid in known:
                    edges["memberof"].append((id_to_idx[m_sid], target_idx))

            # ── ChildObjects: parent → child ──
            for child in obj.get("ChildObjects", []) or []:
                c_sid = (child if isinstance(child, str)
                         else child.get("ObjectIdentifier", "")).upper()
                if c_sid in known:
                    edges["contains"].append((target_idx, id_to_idx[c_sid]))

            # ── Links (GPO links to OUs/domains): GPO → linked object ──
            for link in obj.get("Links", []) or []:
                l_sid = (link if isinstance(link, str)
                         else link.get("ObjectIdentifier", "")).upper()
                if l_sid in known:
                    edges["gplink"].append((target_idx, id_to_idx[l_sid]))

            # ── Trust edges from domain objects: source → target ──
            if nt == "domains":
                for trust in obj.get("Trusts", []) or []:
                    target_dom_sid = (trust.get("TargetDomainSid") or "").upper()
                    if target_dom_sid not in known:
                        continue
                    rel = trust_type_to_edge_name(trust.get("TrustType", ""))
                    edges[rel].append((target_idx, id_to_idx[target_dom_sid]))
                    # bidirectional trusts: add reverse too
                    if trust.get("TrustDirection") == "Bidirectional":
                        edges[rel].append((id_to_idx[target_dom_sid], target_idx))

    # convert to tensors
    edge_tensors = {}
    for rel, pairs in edges.items():
        if not pairs:
            continue
        t = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        edge_tensors[rel] = t

    return edge_tensors

# ── assemble HeteroData ──────────────────────────────────────────────────────

def build_heterodata(X: torch.Tensor, edge_tensors: dict) -> HeteroData:
    data = HeteroData()
    data["node"].x         = X
    data["node"].num_nodes = X.shape[0]
    for rel, ei in edge_tensors.items():
        data["node", rel, "node"].edge_index = ei
    return data

# ── zoom.csv handling for training mode ──────────────────────────────────────

def resolve(val: str, name_to_id: dict, id_to_idx: dict):
    if not val or not val.strip():
        return None
    v = val.strip()
    if v.upper() in id_to_idx:
        return id_to_idx[v.upper()]
    lower = v.lower()
    oid   = name_to_id.get(lower)
    if oid:
        return id_to_idx.get(oid)
    short = lower.split("@")[0].split(".")[0]
    oid   = name_to_id.get(short)
    if oid:
        return id_to_idx.get(oid)
    return None

def load_zoom(path: Path, name_to_id: dict, id_to_idx: dict,
              X: torch.Tensor) -> tuple[list, set]:
    """
    Parse zoom.csv into training examples.
    Returns (examples_list, objective_idxs_set).
    """
    paths    = defaultdict(list)
    raw_rows = []

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_rows.append(row)
            paths[row["path_id"].strip()].append(row)

    # dynamic objective inference — last next_hop per path
    objective_idxs = set()
    for pid, rows in paths.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["step"]))
        last_hop    = rows_sorted[-1]["next_hop_id"].strip().upper()
        idx         = resolve(last_hop, name_to_id, id_to_idx)
        if idx is not None:
            objective_idxs.add(idx)
            X[idx, -1] = 1.0   # set is_objective feature

    examples = []
    skipped  = 0
    null_idx = X.shape[0] - 1   # Mode 2 sentinel

    for row in raw_rows:
        path_id    = row["path_id"].strip()
        step       = int(row["step"])
        train_on   = int(row.get("train_on_step", "1") or "1")
        path_w     = float(row.get("path_weight", "1.0") or "1.0")
        step_cost  = float(row.get("step_cost",   "0.0") or "0.0")
        eff_weight = path_w * (1.0 - step_cost)

        if train_on == 0:
            continue

        cur  = resolve(row["current_node_id"],    name_to_id, id_to_idx)
        hop  = resolve(row["next_hop_id"],        name_to_id, id_to_idx)
        cred = resolve(row["credential_used_id"], name_to_id, id_to_idx)

        if cur is None or hop is None or cred is None:
            print(f"  WARN [{path_id} step {step}]: unresolved nodes — skipped")
            skipped += 1
            continue

        for mode, cred_idx in [("constrained", cred),
                                ("unconstrained", null_idx)]:
            examples.append({
                "path_id":        path_id,
                "step":           step,
                "current_idx":    cur,
                "credential_idx": cred_idx,
                "next_hop_idx":   hop,
                "eff_weight":     round(eff_weight, 4),
                "mode":           mode,
                "notes":          row.get("notes", ""),
            })

    return examples, objective_idxs, skipped

# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True,
                        help="Path to merged forest_graph.json")
    parser.add_argument("--zoom",  default=None,
                        help="Optional zoom.csv for training mode")
    parser.add_argument("--out",   required=True,
                        help="Output directory")
    args = parser.parse_args()

    graph_path = Path(args.graph)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("build_dataset.py")
    print("=" * 60)
    print(f"Graph : {graph_path}")
    if args.zoom:
        print(f"Zoom  : {args.zoom}")
    print(f"Out   : {out_dir}")
    print()

    # ─ load graph ─
    print("[1] Loading graph...")
    forest, id_to_idx, name_to_id, X = load_graph(graph_path)
    print(f"  Nodes      : {len(id_to_idx)}")
    print(f"  Feature dim: {X.shape[1]}")

    # ─ edges ─
    print("\n[2] Building edges (principal→target convention)...")
    edge_tensors = build_edges(forest, id_to_idx)
    print(f"  Edge types : {len(edge_tensors)}")
    for rel, t in sorted(edge_tensors.items(), key=lambda x: -x[1].shape[1])[:15]:
        print(f"    {rel:<25}: {t.shape[1]}")
    if len(edge_tensors) > 15:
        print(f"    ... and {len(edge_tensors) - 15} more")

    # ─ trust edges report ─
    trust_edges = sum(
        edge_tensors[r].shape[1]
        for r in ["sameforesttrust", "crossforesttrust"]
        if r in edge_tensors
    )
    print(f"\n  Trust edges captured: {trust_edges}")

    # ─ zoom.csv if training mode ─
    objectives = set()
    examples   = []
    if args.zoom:
        print(f"\n[3] Parsing {args.zoom}...")
        examples, objectives, skipped = load_zoom(
            Path(args.zoom), name_to_id, id_to_idx, X
        )
        print(f"  Examples   : {len(examples)}")
        print(f"  Objectives : {len(objectives)}")
        if skipped:
            print(f"  Skipped    : {skipped}")

    # ─ HeteroData ─
    print("\n[4] Assembling HeteroData...")
    data = build_heterodata(X, edge_tensors)
    print(f"  data['node'].x       : {data['node'].x.shape}")
    print(f"  data['node'].num_nodes: {data['node'].num_nodes}")
    print(f"  edge_types           : {len(data.edge_types)}")

    # ─ save ─
    print("\n[5] Saving...")
    hetero_path = out_dir / "heterodata.pt"
    torch.save(data, hetero_path)
    print(f"  heterodata.pt          → {hetero_path}")

    if args.zoom:
        examples_path = out_dir / "training_examples.json"
        examples_path.write_text(
            json.dumps(examples, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"  training_examples.json → {examples_path}")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Nodes      : {len(id_to_idx)}")
    print(f"  Edge types : {len(edge_tensors)}")
    print(f"  Trust edges: {trust_edges}")
    if args.zoom:
        print(f"  Examples   : {len(examples)}")
    print("\nNext: train.py (if examples present) or inference notebook")
    print("=" * 60)

if __name__ == "__main__":
    main()