"""
run_inference.py
----------------
CLI tool that loads a trained model and answers attack-path queries
on a prepared graph. Includes a path quality audit that detects
unreliable outputs and tells the operator honestly when the model
or the data fell short.

Usage:
    python run_inference.py \
        --hetero output/heterodata.pt \
        --graph  output/forest_graph.json \
        --model  models/best_model_gcn.pt \
        --start  "wley@inlanefreight.local" \
        --target "domain admins@inlanefreight.local"

    # use HGT instead (needs checkpoint with metadata)
    python run_inference.py \
        --hetero output/heterodata.pt \
        --graph  output/forest_graph.json \
        --model  models/HGT.pt \
        --model-type hgt \
        --start  "samwell.tarly" \
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
        self.hidden_dim = h  # Save the hidden dimension integer
        self.input_proj = nn.ModuleDict({
            nt: Linear(-1, h) for nt in metadata[0]
        })
        self.convs = nn.ModuleList([
            HGTConv(h, h, metadata, heads=heads) for _ in range(layers)
        ])
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x_dict, ei_dict):
        # Rename 'h' to 'emb_dict' to avoid shadowing the integer
        emb_dict = {k: F.relu(self.input_proj[k](v)) for k, v in x_dict.items()}
        for i, conv in enumerate(self.convs):
            emb_dict = conv(emb_dict, ei_dict)
            if i < len(self.convs) - 1:
                emb_dict = {k: self.dropout(F.relu(v)) for k, v in emb_dict.items()}
        
        # Output is dynamically flattened based on NODE_ORDER to support flat beam search
        out = []
        for nt in NODE_ORDER:
            if nt in emb_dict: 
                out.append(emb_dict[nt])
                
        # Use self.hidden_dim for the empty tensor fallback
        return torch.cat(out, dim=0) if out else torch.zeros((0, self.hidden_dim), device=next(self.parameters()).device)


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
# graph loading + name resolution (Hetero Upgrade)
# ─────────────────────────────────────────────────────────────────────────────

NODE_ORDER = ["users","computers","groups","domains",
               "gpos","ous","containers","cas","certtemplates"]

def build_name_lookup(graph_path: Path, data: HeteroData):
    """
    Builds a unified coordinate system mapping strings to global indices,
    matching the exact concatenation order of the PyG tensors.
    """
    forest = json.loads(graph_path.read_text(encoding="utf-8"))
    
    # Pre-calculate global offsets per node type based on HeteroData tensors
    type_offsets = {}
    curr_offset = 0
    for nt in NODE_ORDER:
        type_offsets[nt] = curr_offset
        if nt in data.node_types:
            curr_offset += data[nt].num_nodes

    name_to_idx = {}
    idx_to_name = {}
    idx_to_type = {}
    
    for nt in NODE_ORDER:
        local_idx = 0
        for obj in forest.get(nt, []) or []:
            oid = (obj.get("ObjectIdentifier") or "").upper()
            if not oid: 
                continue

            name = (obj.get("Properties", {}).get("name") or "unknown").lower()
            global_idx = type_offsets[nt] + local_idx

            # 1. NATIVE SID SUPPORT
            name_to_idx[oid] = global_idx

            # 2. FULL NAME SUPPORT (Protected)
            if name not in name_to_idx:
                name_to_idx[name] = global_idx
                
            # 3. SHORT NAME ALIAS (Protected)
            short = name.split("@")[0].split(".")[0]
            if short and short not in name_to_idx:
                name_to_idx[short] = global_idx

            # Model lookups
            idx_to_name[global_idx] = name
            idx_to_type[global_idx] = nt
            
            local_idx += 1

        # Map the null/sentinel credential appended to "users"
        if nt == "users" and "users" in data.node_types:
            if local_idx < data["users"].num_nodes:
                sentinel_idx = type_offsets["users"] + local_idx
                idx_to_name[sentinel_idx] = "<null>"
                idx_to_type[sentinel_idx] = "sentinel"

    return name_to_idx, idx_to_name, idx_to_type, type_offsets


def find_node(query: str, name_to_idx: dict, idx_to_name: dict):
    q = query.lower().strip()
    if q in name_to_idx: return name_to_idx[q]
    short = q.split("@")[0].split(".")[0]
    if short in name_to_idx: return name_to_idx[short]
    matches = [(i, n) for i, n in idx_to_name.items() if q in n]
    if matches:
        if len(matches) > 1:
            print(f"  Multiple matches for '{query}':")
            for idx, n in matches[:3]: print(f"    [{idx}] {n}")
        return matches[0][0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# edge type catalogue
# ─────────────────────────────────────────────────────────────────────────────

EDGE_TO_TECHNIQUE = {k.lower(): v for k, v in {
    "HasSession":          ("Credential theft (LSASS)", "Extract credentials from active session"),
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
    if "@" in name: return name.split("@", 1)[1]
    if "." in name and not name.startswith("s-"): return name
    return None

def get_neighbors(node_idx, edge_tensors_flat):
    nbrs = set()
    for rel, ei in edge_tensors_flat.items():
        src, dst = ei
        mask = (src == node_idx)
        for n in dst[mask].tolist(): nbrs.add(n)
    nbrs.discard(node_idx)
    return list(nbrs)

def get_edge_type(s, d, edge_tensors_flat):
    for rel, ei in edge_tensors_flat.items():
        src, dst = ei
        if ((src == s) & (dst == d)).any().item(): return rel
    return "unknown"

def has_dcsync_on(node_idx, target_domain_name, edge_tensors_flat, idx_to_name):
    held_on = set()
    for rel in DCSYNC_RIGHTS:
        ei = edge_tensors_flat.get(rel)
        if ei is None: continue
        src, dst = ei
        mask = (src == node_idx)
        for d in dst[mask].tolist(): held_on.add((rel, idx_to_name.get(d, "")))

    has_getchanges    = {d for r, d in held_on if r == "getchanges"}
    has_getchangesall = {d for r, d in held_on if r == "getchangesall"}
    has_filtered      = {d for r, d in held_on if r == "getchangesinfilteredset"}

    domains_with_both = has_getchanges & (has_getchangesall | has_filtered)
    won = False
    for d in domains_with_both:
        if target_domain_name == d or target_domain_name.endswith("." + d):
            won = True; break
    return won, list(domains_with_both)

def beam_search(model, embeddings, edge_tensors_flat, start_idx, target_idx, idx_to_name, edge_name_map, known_edges, beam_width=3, max_depth=8):
    target_domain = get_node_domain(target_idx, idx_to_name) or ""
    beam = [(0.0, [start_idx])]
    completed = []
    
    for _ in range(max_depth):
        candidates = []
        for log_score, path in beam:
            cur = path[-1]
            won, _ = has_dcsync_on(cur, target_domain, edge_tensors_flat, idx_to_name)
            
            if cur == target_idx or won:
                completed.append((log_score, path))
                continue
                
            nbrs = get_neighbors(cur, edge_tensors_flat)
            nbrs = [n for n in nbrs if n not in path and n < embeddings.shape[0]]
            if not nbrs: continue
            
            with torch.no_grad():
                scores = model.score(embeddings, cur, nbrs)
                probs  = F.softmax(scores, dim=0)
                
            for i, nb in enumerate(nbrs):
                # 1. Identify the relationship type
                rel = edge_name_map.get((cur, nb), "unknown")
                
                # 2. THE FALLBACK BYPASS
                # If the model has never seen this edge during training, bypass the 
                # neural network's score and assign it a low baseline operational score.
                if rel not in known_edges:
                    prob_val = 0.05
                else:
                    prob_val = probs[i].item()
                    
                # 3. Calculate the new sequence score
                ns = log_score + math.log(prob_val + 1e-9)
                new_path = path + [nb]
                candidates.append((ns, new_path))
                    
        if not candidates: break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:beam_width]
        if len(completed) >= beam_width: break
        
    result = completed if completed else beam
    result.sort(key=lambda x: x[0], reverse=True)
    
    trimmed = []
    for score, path in result[:beam_width]:
        cut_at = len(path)
        for i, node in enumerate(path):
            won, _ = has_dcsync_on(node, target_domain, edge_tensors_flat, idx_to_name)
            if node == target_idx or won:
                cut_at = i + 1
                break
        trimmed.append((score, path[:cut_at]))
        
    return trimmed


# ─────────────────────────────────────────────────────────────────────────────
# path quality audit
# ─────────────────────────────────────────────────────────────────────────────

def audit_path(path, target_idx, edge_tensors_flat, idx_to_name):
    audit = []
    final, target = path[-1], target_idx
    target_domain = get_node_domain(target, idx_to_name) or ""

    if len(path) <= 1:
        audit.append(("fail", "Path has zero steps — beam search could not progress. Start node likely lacks exploitable edges."))
        return audit

    if final == target:
        audit.append(("ok", "Path reaches the requested target directly."))
    else:
        won, dcsync_domains = has_dcsync_on(final, target_domain, edge_tensors_flat, idx_to_name)
        if won:
            audit.append(("ok", f"Path terminates at node with DCSync on target domain '{target_domain}'. Operational success."))
        elif dcsync_domains:
            audit.append(("warn", f"Path has DCSync on {dcsync_domains}, but '{target_domain}' is not directly reachable. Improvisation required."))
        else:
            audit.append(("fail", f"Path terminates at '{idx_to_name.get(final,'?')}' lacking DCSync and is not the target."))

    final_domain = get_node_domain(final, idx_to_name)
    if final_domain and target_domain and final_domain != target_domain and not any(a[0] == "ok" for a in audit):
        audit.append(("warn", f"Final node is in domain '{final_domain}' but target is in '{target_domain}'."))

    if len(path) >= 3:
        last_edge = get_edge_type(path[-2], path[-1], edge_tensors_flat)
        if last_edge in ("memberof", "contains") and final != target:
            audit.append(("warn", f"Path ends structurally with '{last_edge}' edge. Likely noise from beam search."))

    return audit

def explain_terminal_state(final_idx, target_idx, edge_tensors_flat, idx_to_name):
    if final_idx == target_idx: return None
    target_domain = get_node_domain(target_idx, idx_to_name)
    final_name    = idx_to_name.get(final_idx, f"idx_{final_idx}")

    dcsync_domains = set()
    for rel in DCSYNC_RIGHTS:
        ei = edge_tensors_flat.get(rel)
        if ei is None: continue
        src, dst = ei
        mask = (src == final_idx)
        for d in dst[mask].tolist(): dcsync_domains.add(idx_to_name.get(d, ""))

    if not dcsync_domains: return None

    direct = target_domain in dcsync_domains
    bridges = []
    if not direct:
        for dom in dcsync_domains:
            dom_idx = next((i for i, n in idx_to_name.items() if n == dom), None)
            if dom_idx is None: continue
            for rel in ("sameforesttrust", "crossforesttrust"):
                ei = edge_tensors_flat.get(rel)
                if ei is None: continue
                src, dst = ei
                mask = (src == dom_idx)
                for d in dst[mask].tolist():
                    if idx_to_name.get(d, "") == target_domain: bridges.append((dom, rel))

    msg = ["", "  ┌─ TERMINAL STATE REACHED ────────────────────────────────"]
    msg.append(f"  │  '{final_name}'")
    msg.append(f"  │  holds DCSync on: {sorted(dcsync_domains)}")

    if direct:
        msg.append(f"  │\n  │  Direct DA in target domain '{target_domain}'.")
        msg.append(f"  │  Next: secretsdump.py -just-dc <user>:<pw>@<DC>")
    elif bridges:
        msg.append(f"  │\n  │  Target '{target_domain}' reachable via trust:")
        for d, r in bridges: msg.append(f"  │    {d} --[{r}]-> {target_domain}")
        msg.append(f"  │\n  │  Recommended: SID History attack")
        msg.append(f"  │    1. DCSync Krbtgt -> 2. lookupsid Enterprise Admins -> 3. ticketer -> 4. psexec")
    else:
        msg.append(f"  │\n  │  Achieved local DA. No direct trust path to '{target_domain}'. Improvise.")
    msg.append("  └──────────────────────────────────────────────────────────")
    return "\n".join(msg)


# ─────────────────────────────────────────────────────────────────────────────
# pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def print_path(path, score, edge_tensors_flat, idx_to_name, prefix=""):
    prob = math.exp(score)
    print(f"{prefix}Score: {prob:.4f}  Steps: {len(path)-1}")
    for i in range(len(path) - 1):
        s, d = path[i], path[i+1]
        edge = get_edge_type(s, d, edge_tensors_flat)
        tech, desc = EDGE_TO_TECHNIQUE.get(edge.lower(), ("Unknown edge", "Manual review"))
        print(f"{prefix}  [{i+1}] {idx_to_name.get(s, f'idx_{s}')}")
        print(f"{prefix}      → {idx_to_name.get(d, f'idx_{d}')}")
        print(f"{prefix}      edge   : {edge}")
        print(f"{prefix}      attack : {tech} ({desc})")

def print_audit(audit, prefix=""):
    if not audit: return
    print(f"\n{prefix}AUDIT:")
    for level, msg in audit:
        icon = {"ok": "✓", "warn": "⚠", "fail": "✗"}[level]
        print(f"{prefix}  {icon} {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_gcn(checkpoint_path: Path, X_flat: torch.Tensor, edge_index_flat, device):
    model = GCNNavigator(X_flat.shape[1], 64, 32).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad(): emb = model.encode(X_flat, edge_index_flat)
    return model, emb


def load_hgt(checkpoint_path: Path, data: HeteroData, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "metadata" not in ckpt: raise RuntimeError("HGT checkpoint missing 'metadata' key")
    
    training_metadata = (list(ckpt["metadata"][0]), [tuple(et) for et in ckpt["metadata"][1]])
    
    # Pad missing node types with zero tensors to satisfy PyG metadata requirements
    x_dict = {}
    for nt in training_metadata[0]:
        if nt in data.node_types: x_dict[nt] = data[nt].x.to(device)
        else: x_dict[nt] = torch.zeros((0, ckpt.get("num_features", data["users"].x.shape[1])), device=device)
            
    empty = torch.zeros((2, 0), dtype=torch.long, device=device)
    current_edge_types = set(tuple(et) for et in data.edge_types)
    edge_index_dict_training = {
        et: data[et].edge_index.to(device) if et in current_edge_types else empty
        for et in training_metadata[1]
    }

    model = HGTNavigator(
        in_c=ckpt.get("num_features", data["users"].x.shape[1]),
        h=ckpt.get("hidden_dim", 64),
        heads=ckpt.get("num_heads", 4),
        layers=ckpt.get("num_layers", 2),
        p=32,
        metadata=training_metadata,
    ).to(device)
    
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.known_edges = [et[1] for et in training_metadata[1]]
    with torch.no_grad(): emb = model.encode(x_dict, edge_index_dict_training)
    return model, emb


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hetero", required=True)
    parser.add_argument("--graph",  required=True)
    parser.add_argument("--model",  required=True)
    parser.add_argument("--model-type", default="gcn", choices=["gcn", "hgt"])
    parser.add_argument("--start",  required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print("=" * 60)
    # print("GNN-AD-Navigator Inference")
    # print("=" * 60)

    # ─ Graph Loading & Global Index Translation ─
    data = torch.load(args.hetero, map_location=device, weights_only=False)
    name_to_idx, idx_to_name, idx_to_type, offsets = build_name_lookup(Path(args.graph), data)

    # Flatten Hetero Edges to Global Matrix for Beam Search Navigation
    edge_tensors_flat = defaultdict(list)
    if hasattr(data, 'edge_index_dict'):
        for (src_t, rel, dst_t), ei in data.edge_index_dict.items():
            if src_t not in offsets or dst_t not in offsets: continue
            ei_shifted = ei.clone().to(device)
            ei_shifted[0] += offsets[src_t]
            ei_shifted[1] += offsets[dst_t]
            edge_tensors_flat[rel].append(ei_shifted)
    else:
        # Fallback for old homogenous format
        for rel in data.edge_types:
            edge_tensors_flat[rel[1]].append(data[rel].edge_index.to(device))

    for rel in edge_tensors_flat:
        edge_tensors_flat[rel] = torch.cat(edge_tensors_flat[rel], dim=1)

    s_idx = find_node(args.start, name_to_idx, idx_to_name)
    t_idx = find_node(args.target, name_to_idx, idx_to_name)
    if s_idx is None: print(f"ERROR: cannot resolve start '{args.start}'"); sys.exit(1)
    if t_idx is None: print(f"ERROR: cannot resolve target '{args.target}'"); sys.exit(1)

    print(f"Start  : {idx_to_name[s_idx]} ({idx_to_type.get(s_idx,'?')})")
    print(f"Target : {idx_to_name[t_idx]} ({idx_to_type.get(t_idx,'?')})")

    # ─ Model Init ─
    if args.model_type == "gcn":
        # Dynamic flattening for legacy GCN
        X_flat = torch.cat([data[nt].x for nt in NODE_ORDER if nt in data.node_types], dim=0).to(device)
        edge_index_flat = torch.unique(torch.cat(list(edge_tensors_flat.values()), dim=1), dim=1)
        model, emb = load_gcn(Path(args.model), X_flat, edge_index_flat, device)
    else:
        model, emb = load_hgt(Path(args.model), data, device)

# ─ Execute ─
    
    # 1. Extract known edges safely (Pulling directly from the HGTConv layer)
    known_edges = []
    if args.model_type == "hgt":
        known_edges = model.known_edges

    # 2. Build edge_name_map dynamically from the PyG data object
    edge_name_map = {}
    if hasattr(data, 'edge_index_dict'):
        for (src_t, rel, dst_t), ei in data.edge_index_dict.items():
            if src_t not in offsets or dst_t not in offsets: 
                continue
            src_indices = ei[0] + offsets[src_t]
            dst_indices = ei[1] + offsets[dst_t]
            for s, d in zip(src_indices.tolist(), dst_indices.tolist()):
                if (s, d) not in edge_name_map:
                    edge_name_map[(s, d)] = rel
    else:
        # Legacy fallback for old homogenous format
        for rel in data.edge_types:
            ei = data[rel].edge_index
            for s, d in zip(ei[0].tolist(), ei[1].tolist()):
                if (s, d) not in edge_name_map:
                    edge_name_map[(s, d)] = rel[1]

    # 3. Call beam_search with ALL the correct arguments in order!
    paths = beam_search(
        model, 
        emb, 
        edge_tensors_flat, 
        s_idx, 
        t_idx, 
        idx_to_name, 
        edge_name_map,      # <-- Newly added
        known_edges,        # <-- Newly added
        args.beam_width, 
        args.max_depth
    )

    print(f"\n{'=' * 60}\nRESULTS — best scored {len(paths)} path(s)\n{'=' * 60}")
    if not paths:
        print("\nNo paths found.")
        sys.exit(0)

    for rank, (score, path) in enumerate(paths, 1):
        print(f"\nPath {rank}:")
        print_path(path, score, edge_tensors_flat, idx_to_name, prefix="  ")
        print_audit(audit_path(path, t_idx, edge_tensors_flat, idx_to_name), prefix="  ")
        advisory = explain_terminal_state(path[-1], t_idx, edge_tensors_flat, idx_to_name)
        if advisory: print(advisory)

if __name__ == "__main__":
    main()