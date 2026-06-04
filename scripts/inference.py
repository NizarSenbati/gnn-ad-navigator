"""
run_inference.py
----------------
CLI tool that loads a trained model and answers attack-path queries
on a prepared graph. Includes a path quality audit that detects
unreliable outputs and tells the operator honestly when the model
or the data fell short.

Usage:
    python run_inference.py \\
        --hetero output/heterodata.pt \\
        --graph  output/forest_graph.json \\
        --model  models/best_model_gcn.pt \\
        --start  "wley@inlanefreight.local" \\
        --target "domain admins@inlanefreight.local"

    # use HGT instead (needs checkpoint with metadata)
    python run_inference.py \\
        --hetero output/heterodata.pt \\
        --graph  output/forest_graph.json \\
        --model  models/HGT.pt \\
        --model-type hgt \\
        --start  "samwell.tarly" \\
        --target "inlanefreight-ca"
"""

import json
import math
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, HGTConv, Linear

# ─────────────────────────────────────────────────────────────────────────────
# model architectures (must match training)
# ─────────────────────────────────────────────────────────────────────────────

class PolicyHead(nn.Module):
    def __init__(self, h, p):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(h, p), nn.ReLU(), nn.Dropout(0.2), nn.Linear(p, 1)
        )
    def forward(self, cur, nbr):
        return self.mlp(cur.unsqueeze(0).expand_as(nbr) * nbr).squeeze(-1)


class GCNEncoder(nn.Module):
    def __init__(self, in_c, h):
        super().__init__()
        self.conv1 = GCNConv(in_c, h); self.conv2 = GCNConv(h, h)
        self.bn1   = nn.BatchNorm1d(h); self.bn2   = nn.BatchNorm1d(h)
    def forward(self, x, ei):
        x = F.relu(self.bn1(self.conv1(x, ei)))
        x = F.dropout(x, 0.3, self.training)
        return F.relu(self.bn2(self.conv2(x, ei)))


class GCNNavigator(nn.Module):
    def __init__(self, in_c, h, p):
        super().__init__()
        self.encoder     = GCNEncoder(in_c, h)
        self.policy_head = PolicyHead(h, p)
    def encode(self, x, ei): return self.encoder(x, ei)
    def score(self, emb, cur, nbrs):
        return self.policy_head(emb[cur],
               emb[torch.tensor(nbrs, device=emb.device)])


class HGTEncoder(nn.Module):
    def __init__(self, in_c, h, heads, layers, metadata):
        super().__init__()
        self.input_proj = nn.ModuleDict({
            nt: Linear(-1, h) for nt in metadata[0]
        })
        self.convs = nn.ModuleList([
            HGTConv(h, h, metadata, heads=heads) for _ in range(layers)
        ])
        self.dropout = nn.Dropout(0.3)
    def forward(self, x_dict, ei_dict):
        h = {k: F.relu(self.input_proj[k](v)) for k, v in x_dict.items()}
        for i, conv in enumerate(self.convs):
            h = conv(h, ei_dict)
            if i < len(self.convs) - 1:
                h = {k: self.dropout(F.relu(v)) for k, v in h.items()}
        return torch.cat([h[nt] for nt in ["node"] if nt in h], dim=0)


class HGTNavigator(nn.Module):
    def __init__(self, in_c, h, heads, layers, p, metadata):
        super().__init__()
        self.encoder     = HGTEncoder(in_c, h, heads, layers, metadata)
        self.policy_head = PolicyHead(h, p)
    def encode(self, x_dict, ei_dict): return self.encoder(x_dict, ei_dict)
    def score(self, emb, cur, nbrs):
        return self.policy_head(emb[cur],
               emb[torch.tensor(nbrs, device=emb.device)])


# ─────────────────────────────────────────────────────────────────────────────
# graph loading + name resolution
# ─────────────────────────────────────────────────────────────────────────────

NODE_ORDER = ["users","computers","groups","domains",
               "gpos","ous","containers","cas","certtemplates"]

def build_name_lookup(graph_path: Path):
    forest = json.loads(graph_path.read_text(encoding="utf-8"))
    name_to_idx = {}
    idx_to_name = {}
    idx_to_type = {}
    i = 0
    for nt in NODE_ORDER:
        for obj in forest.get(nt, []) or []:
            oid = (obj.get("ObjectIdentifier") or "").upper()
            if not oid:
                continue
            name = (obj.get("Properties", {}).get("name") or "").lower()
            name_to_idx[name]  = i
            idx_to_name[i]     = name
            idx_to_type[i]     = nt
            short = name.split("@")[0].split(".")[0]
            if short and short not in name_to_idx:
                name_to_idx[short] = i
            i += 1
    idx_to_name[i] = "<null>"
    idx_to_type[i] = "sentinel"
    return name_to_idx, idx_to_name, idx_to_type


def find_node(query: str, name_to_idx: dict, idx_to_name: dict):
    q = query.lower().strip()
    if q in name_to_idx: return name_to_idx[q]
    short = q.split("@")[0].split(".")[0]
    if short in name_to_idx: return name_to_idx[short]
    matches = [(i, n) for i, n in idx_to_name.items() if q in n]
    if matches:
        if len(matches) > 1:
            print(f"  Multiple matches for '{query}':")
            for idx, n in matches[:3]:
                print(f"    [{idx}] {n}")
        return matches[0][0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# edge type catalogue
# ─────────────────────────────────────────────────────────────────────────────

EDGE_TO_TECHNIQUE = {k.lower(): v for k, v in {
    "DCSync":              ("DCSync", "Replicate domain credentials"),
    "GetChanges":          ("Partial DCSync", "Replicate non-credential changes"),
    "GetChangesAll":       ("DCSync complete", "Full credential replication"),
    "GenericAll":          ("ACL abuse — full control", "Reset password, add SPN, modify object"),
    "GenericWrite":        ("ACL abuse — targeted write", "Modify SPN, msDS-AllowedToActOnBehalfOfOtherIdentity"),
    "WriteDACL":           ("ACL abuse — DACL modification", "Grant yourself GenericAll on target"),
    "WriteOwner":          ("Take ownership", "Take ownership then modify DACL"),
    "ForceChangePassword": ("Password reset", "Force-change target's password"),
    "AllExtendedRights":   ("All Extended Rights", "Read LAPS, change password, replicate"),
    "AddKeyCredentialLink":("Shadow Credentials", "Add key credential, retrieve via PKINIT"),
    "AllowedToDelegate":   ("Constrained Delegation", "S4U2Self + S4U2Proxy impersonation"),
    "AllowedToAct":        ("RBCD", "Resource-Based Constrained Delegation"),
    "Enroll":              ("ADCS — certificate enrollment", "Request certificate from template"),
    "HttpEnroll":          ("ADCS — ESC8 NTLM relay", "Coerce auth and relay to /certsrv/"),
    "PublishedTo":         ("Template↔CA relationship", "Template is on this CA"),
    "SameForestTrust":     ("Same-forest trust traversal", "Cross domain via SID History"),
    "CrossForestTrust":    ("Cross-forest trust traversal", "Inter-forest pivot"),
    "MemberOf":            ("Group inheritance", "Inherit rights via group membership"),
    "Owns":                ("Object ownership", "Modify DACL of owned object"),
    "Contains":            ("Container hierarchy", "Object lives inside OU/container"),
}.items()}


# ─────────────────────────────────────────────────────────────────────────────
# inference helpers
# ─────────────────────────────────────────────────────────────────────────────

DCSYNC_RIGHTS = {"getchanges", "getchangesall", "getchangesinfilteredset"}


def get_node_domain(node_idx, idx_to_name):
    name = idx_to_name.get(node_idx, "")
    if "@" in name:
        return name.split("@", 1)[1]
    if "." in name and not name.startswith("s-"):
        return name
    return None


def get_neighbors(node_idx, edge_tensors):
    nbrs = set()
    for rel, ei in edge_tensors.items():
        src, dst = ei
        mask = (src == node_idx)
        for n in dst[mask].tolist():
            nbrs.add(n)
    nbrs.discard(node_idx)
    return list(nbrs)


def get_edge_type(s, d, edge_tensors):
    for rel, ei in edge_tensors.items():
        src, dst = ei
        if ((src == s) & (dst == d)).any().item():
            return rel
    return "unknown"


def has_dcsync_on(node_idx, target_domain_name, edge_tensors, idx_to_name):
    held_on = set()
    for rel in DCSYNC_RIGHTS:
        ei = edge_tensors.get(rel)
        if ei is None: continue
        src, dst = ei
        mask = (src == node_idx)
        for d in dst[mask].tolist():
            held_on.add((rel, idx_to_name.get(d, "")))

    has_getchanges    = {d for r, d in held_on if r == "getchanges"}
    has_getchangesall = {d for r, d in held_on if r == "getchangesall"}
    has_filtered      = {d for r, d in held_on if r == "getchangesinfilteredset"}

    # Valid DCSync: GetChanges AND (GetChangesAll OR GetChangesInFilteredSet)
    domains_with_both = has_getchanges & (has_getchangesall | has_filtered)

    # Check if we have DCSync on the target domain OR its parent forest root
    # e.g., target is "kingslanding.sevenkingdoms.local", d is "sevenkingdoms.local"
    won = False
    for d in domains_with_both:
        if target_domain_name == d or target_domain_name.endswith("." + d):
            won = True
            break

    return won, list(domains_with_both)


#def beam_search(model, embeddings, edge_tensors, start_idx, target_idx,
#                idx_to_name, beam_width=3, max_depth=8):
#    target_domain = get_node_domain(target_idx, idx_to_name) or ""
#    beam = [(0.0, [start_idx])]
#    completed = []
#    for _ in range(max_depth):
#        candidates = []
#        for log_score, path in beam:
#            cur = path[-1]
#            won, _ = has_dcsync_on(cur, target_domain, edge_tensors, idx_to_name)
#            if cur == target_idx or won:
#                completed.append((log_score, path))
#                continue
#            nbrs = get_neighbors(cur, edge_tensors)
#            nbrs = [n for n in nbrs if n not in path]
#            if not nbrs: continue
#            with torch.no_grad():
#                scores = model.score(embeddings, cur, nbrs)
#                probs  = F.softmax(scores, dim=0)
#            for i, nb in enumerate(nbrs):
#                ns = log_score + math.log(probs[i].item() + 1e-9)
#                new_path = path + [nb]
#                candidates.append((ns, new_path))
#                won_nb, _ = has_dcsync_on(nb, target_domain, edge_tensors, idx_to_name)
#                if nb == target_idx or won_nb:
#                    completed.append((ns, new_path))
#        if not candidates: break
#        candidates.sort(key=lambda x: x[0], reverse=True)
#        beam = candidates[:beam_width]
#        if len(completed) >= beam_width: break
#    
#    result = completed if completed else beam
#    result.sort(key=lambda x: x[0], reverse=True)
#    
#    # NEW — trim each path at the earliest DCSync-capable node
#    trimmed = []
#    for score, path in result[:beam_width]:
#        cut_at = len(path)
#        for i, node in enumerate(path):
#            won, _ = has_dcsync_on(node, target_domain, edge_tensors, idx_to_name)
#            if node == target_idx or won:
#                cut_at = i + 1
#                break
#        trimmed.append((score, path[:cut_at]))
#    
#    return trimmed

def beam_search(model, embeddings, edge_tensors, start_idx, target_idx, idx_to_name, beam_width=3, max_depth=8):
    target_domain = get_node_domain(target_idx, idx_to_name) or ""
    beam = [(0.0, [start_idx])]
    completed = []
    
    for _ in range(max_depth):
        candidates = []
        for log_score, path in beam:
            cur = path[-1]
            won, _ = has_dcsync_on(cur, target_domain, edge_tensors, idx_to_name)
            
            if cur == target_idx or won:
                completed.append((log_score, path))
                continue
                
            nbrs = get_neighbors(cur, edge_tensors)
            nbrs = [n for n in nbrs if n not in path]
            if not nbrs: continue
            
            with torch.no_grad():
                scores = model.score(embeddings, cur, nbrs)
                probs  = F.softmax(scores, dim=0)
                
            for i, nb in enumerate(nbrs):
                ns = log_score + math.log(probs[i].item() + 1e-9)
                new_path = path + [nb]
                candidates.append((ns, new_path))
                
                won_nb, _ = has_dcsync_on(nb, target_domain, edge_tensors, idx_to_name)
                if nb == target_idx or won_nb:
                    completed.append((ns, new_path))
                    
        if not candidates: break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:beam_width]
        if len(completed) >= beam_width: break
        
    result = completed if completed else beam
    result.sort(key=lambda x: x[0], reverse=True)
    
    # POST-PROCESSING TRIM: Force cut the path at the earliest terminal state
    trimmed = []
    for score, path in result[:beam_width]:
        cut_at = len(path)
        for i, node in enumerate(path):
            won, _ = has_dcsync_on(node, target_domain, edge_tensors, idx_to_name)
            if node == target_idx or won:
                cut_at = i + 1  # Keep the winning node, discard trailing noise
                break
        trimmed.append((score, path[:cut_at]))
        
    return trimmed

# def beam_search(model, embeddings, edge_tensors, start_idx, target_idx, idx_to_name, beam_width=3, max_depth=8):
#     target_domain = get_node_domain(target_idx, idx_to_name) or ""
#     beam = [(0.0, [start_idx])]
#     completed = []
#     for _ in range(max_depth):
#         candidates = []
#         for log_score, path in beam:
#             cur = path[-1]
#             won, _ = has_dcsync_on(cur, target_domain, edge_tensors, idx_to_name)
#             if cur == target_idx or won:
#                 completed.append((log_score, path))
#                 continue
#             nbrs = get_neighbors(cur, edge_tensors)
#             nbrs = [n for n in nbrs if n not in path]
#             if not nbrs: continue
#             with torch.no_grad():
#                 scores = model.score(embeddings, cur, nbrs)
#                 probs  = F.softmax(scores, dim=0)
#             for i, nb in enumerate(nbrs):
#                 ns = log_score + math.log(probs[i].item() + 1e-9)
#                 new_path = path + [nb]
#                 candidates.append((ns, new_path))
#                 won_nb, _ = has_dcsync_on(nb, target_domain, edge_tensors, idx_to_name)
#                 if nb == target_idx or won_nb:
#                     completed.append((ns, new_path))
#         if not candidates: break
#         candidates.sort(key=lambda x: x[0], reverse=True)
#         beam = candidates[:beam_width]
#         if len(completed) >= beam_width: break
#     result = completed if completed else beam
#     result.sort(key=lambda x: x[0], reverse=True)
#     return result[:beam_width]


# ─────────────────────────────────────────────────────────────────────────────
# path quality audit — the heart of "tell the user honestly"
# ─────────────────────────────────────────────────────────────────────────────

def audit_path(path, target_idx, edge_tensors, idx_to_name):
    """
    Returns a list of (level, message) tuples flagging unreliable outputs.
    Levels: 'ok', 'warn', 'fail'.
    """
    audit = []
    final  = path[-1]
    target = target_idx
    target_domain = get_node_domain(target, idx_to_name) or ""

    # check 1: path length
    if len(path) <= 1:
        audit.append(("fail",
            "Path has zero steps — beam search could not progress. "
            "Likely the start node has no outgoing exploit edges."))
        return audit

    # check 2: target reachability
    if final == target:
        audit.append(("ok", "Path reaches the requested target directly."))
    else:
        won, dcsync_domains = has_dcsync_on(
            final, target_domain, edge_tensors, idx_to_name)
        if won:
            audit.append(("ok",
                f"Path terminates at a node with DCSync on the target domain "
                f"'{target_domain}'. This is operational success."))
        elif dcsync_domains:
            audit.append(("warn",
                f"Path terminates with DCSync on {dcsync_domains}, but target "
                f"domain '{target_domain}' is not directly reachable. "
                f"Operator must improvise (trust pivot, SID history, etc.)."))
        else:
            audit.append(("fail",
                f"Path terminates at '{idx_to_name.get(final,'?')}' "
                f"which has no DCSync rights and is not the target. "
                f"The model could not find a working attack chain."))

    # check 3: final-step domain match
    final_domain = get_node_domain(final, idx_to_name)
    if final_domain and target_domain and final_domain != target_domain:
        if not any(a[0] == "ok" for a in audit):
            audit.append(("warn",
                f"Final node is in domain '{final_domain}' but target is in "
                f"'{target_domain}'. Cross-domain attack may need additional steps."))

    # check 4: trailing memberof / contains noise
    if len(path) >= 3:
        last_edge = get_edge_type(path[-2], path[-1], edge_tensors)
        if last_edge in ("memberof", "contains") and final != target:
            audit.append(("warn",
                f"Path ends with a '{last_edge}' edge — this is structural, "
                f"not exploitative. Likely noise from beam search continuation."))

    return audit


def explain_terminal_state(final_idx, target_idx, edge_tensors, idx_to_name):
    """
    If beam search stopped at a DCSync-capable node, explain to the operator
    what's been achieved and how to convert it into the requested objective.
    """
    if final_idx == target_idx:
        return None

    target_domain = get_node_domain(target_idx, idx_to_name)
    final_name    = idx_to_name.get(final_idx, f"idx_{final_idx}")

    dcsync_domains = set()
    for rel in DCSYNC_RIGHTS:
        ei = edge_tensors.get(rel)
        if ei is None: continue
        src, dst = ei
        mask = (src == final_idx)
        for d in dst[mask].tolist():
            dcsync_domains.add(idx_to_name.get(d, ""))

    if not dcsync_domains:
        return None

    direct = target_domain in dcsync_domains
    bridges = []
    if not direct:
        for dom in dcsync_domains:
            dom_idx = next((i for i, n in idx_to_name.items() if n == dom), None)
            if dom_idx is None: continue
            for rel in ("sameforesttrust", "crossforesttrust"):
                ei = edge_tensors.get(rel)
                if ei is None: continue
                src, dst = ei
                mask = (src == dom_idx)
                for d in dst[mask].tolist():
                    if idx_to_name.get(d, "") == target_domain:
                        bridges.append((dom, rel))

    msg = ["", "  ┌─ TERMINAL STATE REACHED ────────────────────────────────"]
    msg.append(f"  │  '{final_name}'")
    msg.append(f"  │  holds DCSync on: {sorted(dcsync_domains)}")

    if direct:
        msg.append(f"  │")
        msg.append(f"  │  Direct DA in target domain '{target_domain}'.")
        msg.append(f"  │  Next: secretsdump.py -just-dc <user>:<pw>@<DC>")
        msg.append(f"  │        → recover any user's hash, including target group members")
    elif bridges:
        msg.append(f"  │")
        msg.append(f"  │  Target '{target_domain}' reachable via trust:")
        for d, r in bridges:
            msg.append(f"  │    {d} --[{r}]-> {target_domain}")
        msg.append(f"  │")
        msg.append(f"  │  Recommended: SID History attack")
        msg.append(f"  │    1. DCSync current domain → child krbtgt hash")
        msg.append(f"  │    2. lookupsid.py → parent Enterprise Admins SID")
        msg.append(f"  │    3. ticketer.py -extra-sid <parent EA SID>")
        msg.append(f"  │    4. psexec.py with forged ticket → SYSTEM on parent DC")
    else:
        msg.append(f"  │")
        msg.append(f"  │  Achieved local DA in {sorted(dcsync_domains)}.")
        msg.append(f"  │  No direct trust path to '{target_domain}' detected.")
        msg.append(f"  │  Operator must improvise: check foreign group memberships,")
        msg.append(f"  │  cross-forest Kerberoasting, or relay attacks.")
    msg.append("  └──────────────────────────────────────────────────────────")
    return "\n".join(msg)


# ─────────────────────────────────────────────────────────────────────────────
# pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def print_path(path, score, edge_tensors, idx_to_name, prefix=""):
    prob = math.exp(score)
    print(f"{prefix}Score: {prob:.4f}  Steps: {len(path)-1}")
    for i in range(len(path) - 1):
        s, d = path[i], path[i+1]
        edge = get_edge_type(s, d, edge_tensors)
        tech, desc = EDGE_TO_TECHNIQUE.get(edge.lower(),
                                            ("Unknown edge", "Manual review"))
        s_name = idx_to_name.get(s, f"idx_{s}")
        d_name = idx_to_name.get(d, f"idx_{d}")
        print(f"{prefix}  [{i+1}] {s_name}")
        print(f"{prefix}      → {d_name}")
        print(f"{prefix}      edge   : {edge}")
        print(f"{prefix}      attack : {tech}")
        print(f"{prefix}      action : {desc}")


def print_audit(audit, prefix=""):
    if not audit: return
    print(f"\n{prefix}AUDIT:")
    for level, msg in audit:
        icon = {"ok": "✓", "warn": "⚠", "fail": "✗"}[level]
        print(f"{prefix}  {icon} {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_gcn(checkpoint_path: Path, X: torch.Tensor, edge_index_flat,
             device):
    HIDDEN_DIM = 64; POLICY_DIM = 32
    model = GCNNavigator(X.shape[1], HIDDEN_DIM, POLICY_DIM).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        emb = model.encode(X, edge_index_flat)
    return model, emb


def load_hgt(checkpoint_path: Path, X: torch.Tensor, data: HeteroData, device):
    HIDDEN_DIM = 64; POLICY_DIM = 32; NUM_HEADS = 4; NUM_LAYERS = 2
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "metadata" not in ckpt:
        raise RuntimeError("HGT checkpoint missing 'metadata' key — "
                            "this script requires HGT.pt")
    training_metadata = (
        list(ckpt["metadata"][0]),
        [tuple(et) for et in ckpt["metadata"][1]],
    )
    current_edge_types = set(tuple(et) for et in data.edge_types)
    empty = torch.zeros((2, 0), dtype=torch.long, device=device)
    edge_index_dict_training = {
        et: data[et].edge_index if et in current_edge_types else empty
        for et in training_metadata[1]
    }

    model = HGTNavigator(
        in_c=ckpt.get("num_features", X.shape[1]),
        h=ckpt.get("hidden_dim", HIDDEN_DIM),
        heads=ckpt.get("num_heads", NUM_HEADS),
        layers=ckpt.get("num_layers", NUM_LAYERS),
        p=POLICY_DIM,
        metadata=training_metadata,
    ).to(device)
    x_dict = {"node": X}
    with torch.no_grad():
        _ = model.encode(x_dict, edge_index_dict_training)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        emb = model.encode(x_dict, edge_index_dict_training)
    return model, emb


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hetero", required=True, help="Path to heterodata.pt")
    parser.add_argument("--graph",  required=True, help="Path to forest_graph.json")
    parser.add_argument("--model",  required=True, help="Path to model checkpoint")
    parser.add_argument("--model-type", default="gcn", choices=["gcn", "hgt"])
    parser.add_argument("--start",  required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("GNN-AD-Navigator inference")
    print("=" * 60)
    print(f"Graph  : {args.graph}")
    print(f"Hetero : {args.hetero}")
    print(f"Model  : {args.model}  ({args.model_type.upper()})")
    print(f"Device : {device}")

    # load graph & lookups
    name_to_idx, idx_to_name, idx_to_type = build_name_lookup(Path(args.graph))
    data = torch.load(args.hetero, map_location=device, weights_only=False)
    X = data["node"].x.to(device)

    # build flat edge index for GCN
    all_edges = torch.cat([data[et].edge_index.cpu()
                            for et in data.edge_types], dim=1)
    edge_index_flat = torch.unique(all_edges, dim=1).to(device)

    edge_tensors = {et[1]: data[et].edge_index.to(device)
                    for et in data.edge_types}

    # resolve nodes
    s_idx = find_node(args.start,  name_to_idx, idx_to_name)
    t_idx = find_node(args.target, name_to_idx, idx_to_name)
    if s_idx is None:
        print(f"ERROR: could not resolve start '{args.start}'"); sys.exit(1)
    if t_idx is None:
        print(f"ERROR: could not resolve target '{args.target}'"); sys.exit(1)

    print(f"\nStart  : {idx_to_name[s_idx]} ({idx_to_type.get(s_idx,'?')})")
    print(f"Target : {idx_to_name[t_idx]} ({idx_to_type.get(t_idx,'?')})")

    # load model
    if args.model_type == "gcn":
        model, emb = load_gcn(Path(args.model), X, edge_index_flat, device)
    else:
        model, emb = load_hgt(Path(args.model), X, data, device)

    # run beam search
    paths = beam_search(model, emb, edge_tensors, s_idx, t_idx,
                        idx_to_name, args.beam_width, args.max_depth)

    print(f"\n{'=' * 60}")
    print(f"RESULTS — top {len(paths)} path(s)")
    print("=" * 60)

    if not paths:
        print("\nNo paths found. The model could not navigate from start to "
              "target within the allowed depth.")
        sys.exit(0)

    for rank, (score, path) in enumerate(paths, 1):
        print(f"\nPath {rank}:")
        print_path(path, score, edge_tensors, idx_to_name, prefix="  ")
        audit = audit_path(path, t_idx, edge_tensors, idx_to_name)
        print_audit(audit, prefix="  ")
        advisory = explain_terminal_state(path[-1], t_idx, edge_tensors, idx_to_name)
        if advisory:
            print(advisory)

    # global verdict
    any_ok   = any(any(a[0] == "ok"   for a in audit_path(p, t_idx, edge_tensors, idx_to_name))
                    for _, p in paths)
    all_fail = all(any(a[0] == "fail" for a in audit_path(p, t_idx, edge_tensors, idx_to_name))
                    for _, p in paths)

    print(f"\n{'=' * 60}")
    if any_ok:
        print("Verdict: at least one path is operationally valid.")
    elif all_fail:
        print("Verdict: ALL paths failed quality audit.")
        print("Possible causes:")
        print("  - Target unreachable from start in the graph")
        print("  - Collection may be incomplete (missing trust edges, foreign memberships)")
        print("  - Start node has no exploitable outgoing rights")
    else:
        print("Verdict: paths require operator improvisation — see advisories above.")
    print("=" * 60)


if __name__ == "__main__":
    main()