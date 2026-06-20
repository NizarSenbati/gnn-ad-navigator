"""
stitching.py
------------
Injects ADCS data from a Certipy scan into a single-domain forest fragment.

Pipeline position:
    raw bloodhound scan
        → minimal_filter.py        (drops dead nodes)
        → stitching.py             ← THIS SCRIPT
            • adds CA + template nodes with synthetic SIDs
            • adds Enroll / PublishedTo / WriteDACL / WriteOwner / GenericAll edges
            • adds HttpEnroll edges for ESC8
        → merger.py                (combines all domains into one forest)
        → build_dataset.py         (produces heterodata.pt)

INPUTS
    --certipy   Certipy JSON output (e.g. 20260504174452_Certipy.json)
    --domain    target domain name (e.g. essos.local, inlanefreight.local)
    --input     one of:
                  (a) folder of cleaned bloodhound JSON files
                  (b) a single forest_graph.json file (prepared single-domain)
    --output    where to write the stitched output

OUTPUTS
    a stitched single-domain forest JSON file containing:
        the original users / computers / groups / domains / gpos / ous / containers
        + cas             (new node type)
        + certtemplates   (new node type)
        + new edges injected as Aces entries on appropriate principals

Usage:
    # from a cleaned folder
    python stitching.py \\
        --certipy data/certipy/20260504174452_Certipy.json \\
        --domain  essos.local \\
        --input   data/high_scans/essos/cleaned/ \\
        --output  data/high_scans/essos/prepared.json

    # from an already-prepared single-domain forest
    python stitching.py \\
        --certipy data/certipy/20260504174452_Certipy.json \\
        --domain  essos.local \\
        --input   data/high_scans/essos/prepared.json \\
        --output  data/high_scans/essos/prepared.json
"""

import json
import hashlib
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import time
from collections import defaultdict, Counter

NODE_TYPES = ["users", "computers", "groups", "domains",
              "gpos", "ous", "containers", "cas", "certtemplates"]


# ── input loading: folder or single file ─────────────────────────────────────

def is_forest_graph_json(path: Path) -> bool:
    """Detect single-file forest_graph.json by top-level keys."""
    try:
        head = json.loads(path.read_text(encoding="utf-8"))
        return any(k in head for k in NODE_TYPES)
    except Exception:
        return False


def load_folder(folder: Path) -> dict:
    """Load all *_<nodetype>.json files in a folder into a forest dict."""
    forest = {nt: [] for nt in NODE_TYPES}
    for f in sorted(folder.glob("*.json")):
        stem = f.stem.lower()
        for nt in NODE_TYPES:
            if stem.endswith(f"_{nt}") or stem == nt:
                raw = json.loads(f.read_text(encoding="utf-8"))
                objs = raw.get("data", []) if isinstance(raw, dict) else []
                forest[nt].extend(objs)
                break
    return forest


def load_forest_file(path: Path) -> dict:
    """Load an existing forest_graph.json file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    forest = {nt: raw.get(nt, []) or [] for nt in NODE_TYPES}
    return forest


def load_input(path: Path) -> dict:
    """Dispatch loader by input type."""
    if path.is_file() and is_forest_graph_json(path):
        return load_forest_file(path)
    if path.is_dir():
        return load_folder(path)
    raise ValueError(f"Unrecognised input: {path} "
                      f"(expected folder or forest_graph.json)")


# ── synthetic SID generation ─────────────────────────────────────────────────

def make_synthetic_sid(name: str, domain: str) -> str:
    """Deterministic synthetic SID per (template_name, domain)."""
    key = f"{name.lower()}@{domain.lower()}"
    h   = hashlib.md5(key.encode()).hexdigest()[:8].upper()
    return f"S-1-5-21-GHOST-{h}"


# ── name → SID resolver from existing forest ────────────────────────────────

def build_resolver(forest: dict) -> dict:
    """Resolve lowercase principal name → (ObjectIdentifier, NodeType)."""
    resolver = {}
    for nt in ["groups", "users", "computers"]:
        for obj in forest.get(nt, []):
            props = obj.get("Properties", {}) or {}
            name  = (props.get("name") or "").lower()
            oid   = (obj.get("ObjectIdentifier") or "").upper()
            if not name or not oid:
                continue
            
            # Store tuple: (SID, Node_Type)
            resolver[name] = (oid, nt) 
            short = name.split("@")[0]
            if short and short not in resolver:
                resolver[short] = (oid, nt)
    return resolver

def resolve_principal(raw_name: str, resolver: dict) -> tuple[str, str] | tuple[None, None]:
    """Returns (ObjectIdentifier, NodeType)."""
    short = raw_name.split("\\")[-1].lower() if "\\" in raw_name else raw_name.lower()
    
    if short in resolver:
        return resolver[short]
    for key in resolver:
        if key.startswith(short + "@"):
            return resolver[key]
    return None, None


# ── certipy parsing ──────────────────────────────────────────────────────────

def parse_certipy(certipy_path: Path) -> tuple[list, list]:
    raw = json.loads(certipy_path.read_text(encoding="utf-8"))
    ca_section   = raw.get("Certificate Authorities", {}) or {}
    tmpl_section = raw.get("Certificate Templates",   {}) or {}
    cas       = [ca_section[k]   for k in sorted(ca_section.keys(),   key=int)]
    templates = [tmpl_section[k] for k in sorted(tmpl_section.keys(), key=int)]
    return cas, templates


# ── node construction ────────────────────────────────────────────────────────

def build_ca_nodes(cas: list, domain: str) -> list:
    nodes = []
    for ca in cas:
        name    = ca.get("CA Name", "unknown-ca")
        dns     = ca.get("DNS Name", "")
        web_enr = ca.get("Web Enrollment", "Disabled") == "Enabled"
        oid     = make_synthetic_sid(name, domain)
        vuln    = ca.get("[!] Vulnerabilities", {}) or {}

        nodes.append({
            "ObjectIdentifier": oid,
            "Properties": {
                "name":              f"{name}@{domain}".lower(),
                "domain":            domain.lower(),
                "distinguishedname": f"cn={name},cn=certification authorities,{dns}".lower(),
                "ca_name":           name,
                "dns_name":          dns,
                "web_enrollment":    web_enr,
                "esc7":              "ESC7" in vuln,
                "esc8":              "ESC8" in vuln,
                "enabled":           True,
                "admincount":        False,
                "is_ca":             True,
            },
            "Aces":        [],
            "IsGhost":     True,
            "GhostReason": "Certipy CA — synthetic SID (no SID in Certipy output)",
        })
    return nodes


def build_template_nodes(templates: list, domain: str) -> list:
    nodes = []
    for tmpl in templates:
        name    = tmpl.get("Template Name", "unknown")
        enabled = tmpl.get("Enabled", False)
        oid     = make_synthetic_sid(name, domain)
        vuln    = tmpl.get("[!] Vulnerabilities", {}) or {}

        nodes.append({
            "ObjectIdentifier": oid,
            "Properties": {
                "name":                      f"{name}@{domain}".lower(),
                "domain":                    domain.lower(),
                "distinguishedname":         f"cn={name},cn=certificate templates".lower(),
                "template_name":             name.lower(),
                "enabled":                   enabled,
                "client_authentication":     tmpl.get("Client Authentication", False),
                "enrollee_supplies_subject": tmpl.get("Enrollee Supplies Subject", False),
                "requires_manager_approval": tmpl.get("Requires Manager Approval", False),
                "authorized_signatures":     tmpl.get("Authorized Signatures Required", 0),
                "any_purpose":               tmpl.get("Any Purpose", False),
                "enrollment_agent":          tmpl.get("Enrollment Agent", False),
                "admincount":                False,
                "hasspn":                    False,
                "esc1":  "ESC1"  in vuln,
                "esc2":  "ESC2"  in vuln,
                "esc3":  "ESC3"  in vuln,
                "esc4":  "ESC4"  in vuln,
                "esc9":  "ESC9"  in vuln,
                "esc13": "ESC13" in vuln,
            },
            "Aces":                  [],
            "EnrollmentRights":      tmpl.get("Permissions", {})
                                          .get("Enrollment Permissions", {})
                                          .get("Enrollment Rights", []),
            "ObjectControlPermissions": tmpl.get("Permissions", {})
                                             .get("Object Control Permissions", {}),
            "CertificateAuthorities": tmpl.get("Certificate Authorities", []),
            "IsGhost":     True,
            "GhostReason": "Certipy template — synthetic SID, real edges",
        })
    return nodes


# ── edge construction ────────────────────────────────────────────────────────
# direction convention: each ACE stored on the TARGET object pointing at
# the principal. build_dataset.py inverts to principal→target convention
# when producing tensors. This keeps the JSON BloodHound-compatible.

def build_adcs_edges(ca_nodes: list, template_nodes: list, resolver: dict, domain_oid: str) -> list:
    edges  = []
    ca_map = {n["Properties"]["ca_name"]: n["ObjectIdentifier"] for n in ca_nodes}

    for tmpl_node in template_nodes:
        tmpl_oid  = tmpl_node["ObjectIdentifier"]

        # Enroll
        for raw_principal in tmpl_node.get("EnrollmentRights", []):
            src_oid, src_type = resolve_principal(raw_principal, resolver)
            if src_oid:
                edges.append({"src": src_oid, "src_type": src_type, "dst": tmpl_oid, "type": "Enroll"})

        # PublishedTo (template → CA)
        for ca_name in tmpl_node.get("CertificateAuthorities", []):
            ca_oid = ca_map.get(ca_name)
            if ca_oid:
                edges.append({"src": tmpl_oid, "src_type": "certtemplates", "dst": ca_oid, "type": "PublishedTo"})

        # Object control (GenericAll, WriteDACL, etc.)
        ocp = tmpl_node.get("ObjectControlPermissions", {}) or {}
        for right, right_name in [("Full Control Principals", "GenericAll"), 
                                  ("Write Dacl Principals", "WriteDACL"), 
                                  ("Write Owner Principals", "WriteOwner")]:
            for raw in ocp.get(right, []):
                src_oid, src_type = resolve_principal(raw, resolver)
                if src_oid:
                    edges.append({"src": src_oid, "src_type": src_type, "dst": tmpl_oid, "type": right_name})

    # ESC8 — Domain Users → CA via HttpEnroll
    domain_users_oid, du_type = resolve_principal("domain users", resolver)
    for ca_node in ca_nodes:
        if ca_node["Properties"].get("esc8") and domain_users_oid:
            edges.append({
                "src": domain_users_oid, 
                "src_type": du_type, 
                "dst": ca_node["ObjectIdentifier"], 
                "type": "HttpEnroll"
            })

    # --- THE MISSING LINK: Escape the CA Black Hole ---
    if domain_oid:
        for ca_node in ca_nodes:
            edges.append({
                "src": ca_node["ObjectIdentifier"],
                "src_type": "cas",
                "dst": domain_oid,
                "type": "TrustedForNTAuth"
            })

    return edges


# ── inject nodes and edges into the forest ──────────────────────────────────

def inject(forest: dict, ca_nodes: list, template_nodes: list,
           edges: list) -> tuple[int, int]:
    """
    Adds ADCS nodes to forest and edges as Aces entries on the appropriate
    source nodes. ACL edges follow BloodHound's storage convention:
    the ACE lives on the TARGET object, with PrincipalSID = source.

    Returns (edges_injected, edges_skipped).
    """
    forest.setdefault("cas",           [])
    forest.setdefault("certtemplates", [])

    forest["cas"].extend(ca_nodes)
    forest["certtemplates"].extend(template_nodes)

    # build sid → object lookup for edge injection
    oid_to_obj = {}
    for objs in forest.values():
        for obj in objs:
            oid = (obj.get("ObjectIdentifier") or "").upper()
            if oid:
                oid_to_obj[oid] = obj

    injected = 0
    skipped  = 0
    for edge in edges:
        # ACE stored on target object, referencing the principal (source)
        target_obj = oid_to_obj.get(edge["dst"])
        if target_obj is None:
            skipped += 1
            continue
        target_obj.setdefault("Aces", []).append({
            "PrincipalSID":  edge["src"],
            "PrincipalType": edge["src_type"].capitalize(), # Now returns 'User', 'Group', 'Cas', etc.
            "RightName":     edge["type"],
            "IsInherited":   False,
            "Source":        "Certipy",
        })
        injected += 1
    return injected, skipped


# ── synthetic edges (coercetgt) ─────────────────────────────────────────────

def build_synthetic_edges(forest: dict) -> list:
    """
    Scans the existing forest to find logically vulnerable delegation paths 
    and generates synthetic 'coercetgt' edges.
    """
    edges = []
    
    # Flatten the forest for easy cross-referencing, keeping track of node types
    all_nodes = []
    for nt, objs in forest.items():
        for obj in objs:
            all_nodes.append((nt, obj))
            
    for src_type, src_node in all_nodes:
        src_props = src_node.get("Properties", {})
        
        # Condition 1: Source has Unconstrained Delegation
        if src_props.get("unconstraineddelegation") == True:
            src_oid = src_node.get("ObjectIdentifier")
            if not src_oid: continue
            
            for tgt_type, tgt_node in all_nodes:
                
                # ---------------------------------------------------------
                # CRITICAL FIX: Target MUST be a Computer account!
                # This prevents wormholing into OUs, Groups, or Domains.
                # ---------------------------------------------------------
                if tgt_type != "computers":
                    continue
                
                tgt_props = tgt_node.get("Properties", {})
                distinguished_name = tgt_props.get("distinguishedname", "").lower()
                
                # Condition 2: Target is a Domain Controller
                if "ou=domain controllers" in distinguished_name:
                    
                    # Condition 3: Target is NOT protected against delegation
                    if tgt_props.get("sensitive") == True:
                        continue
                    
                    tgt_oid = tgt_node.get("ObjectIdentifier")
                    
                    # Prevent a DC from coercing itself
                    if not tgt_oid or src_oid == tgt_oid:
                        continue
                    
                    # Append using the exact format expected by inject()
                    edges.append({
                        "src": src_oid,
                        "src_type": src_type,
                        "dst": tgt_oid,
                        "type": "coercetgt"
                    })
                    
    print(f"[*] Generated {len(edges)} synthetic CoerceToTGT edges.")
    return edges

# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--certipy", required=True,
                        help="Path to Certipy full JSON output")
    parser.add_argument("--domain", required=True,
                        help="Domain name (lowercase, e.g. essos.local)")
    parser.add_argument("--input", required=True,
                        help="Cleaned bloodhound folder OR forest_graph.json file")
    parser.add_argument("--output", required=True,
                        help="Output prepared.json file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    certipy_path = Path(args.certipy)
    input_path   = Path(args.input)
    output_path  = Path(args.output)
    domain       = args.domain.lower()

    if not args.quiet:
        print("=" * 60)
        print("stitching.py")
        print("=" * 60)
        print(f"Certipy : {certipy_path}")
        print(f"Domain  : {domain}")
        print(f"Input   : {input_path}")
        print(f"Output  : {output_path}")
        print()

    # ─ load input ─
    try:
        forest = load_input(input_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        total = sum(len(v) for v in forest.values())
        print(f"Loaded {total} nodes from input")

    # ─ parse certipy ─
    try:
        cas, templates = parse_certipy(certipy_path)
    except Exception as e:
        print(f"ERROR parsing Certipy JSON: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Certipy: {len(cas)} CA(s), {len(templates)} templates")

    # ─ build nodes ─
    ca_nodes       = build_ca_nodes(cas, domain)
    template_nodes = build_template_nodes(templates, domain)

    # ─ resolver + edges ─
    resolver = build_resolver(forest)
    domain_oid = None
    for dom in forest.get("domains", []):
        if dom.get("Properties", {}).get("name", "").lower() == domain:
            domain_oid = dom.get("ObjectIdentifier", "").upper()
            break
    adcs_edges = build_adcs_edges(ca_nodes, template_nodes, resolver, domain_oid)
    
    # 2. Build the synthetic delegation edges
    synthetic_edges = build_synthetic_edges(forest)
    
    # 3. Combine them into a single list for the injector
    edges = adcs_edges + synthetic_edges

    if not args.quiet:
        print(f"Resolver entries     : {len(resolver)}")
        print(f"ADCS edges to inject : {len(adcs_edges)}")
        print(f"CoerceTGT to inject  : {len(synthetic_edges)}")
        print(f"Total edges to inject: {len(edges)}")

    # 4. Inject all of them seamlessly into the forest object
    injected, skipped = inject(forest, ca_nodes, template_nodes, edges)

    if not args.quiet:
        print("\nInjected Edge Inventory:")
        edge_counts = Counter(e["type"] for e in edges)
        for etype, count in sorted(edge_counts.items(), key=lambda x: -x[1]):
            print(f"  {etype:<25}: {count}")
            
        print(f"\n  Total Injected : {injected}")
        print(f"  Total Skipped  : {skipped}  (target node not found)")

    if not args.quiet:
        print(f"  Injected : {injected}")
        print(f"  Skipped  : {skipped}  (target node not found)")

    # ─ write output ─
    output = {nt: forest.get(nt, []) for nt in NODE_TYPES}
    
    # --- ADD THIS METADATA BLOCK ---
    output["_stitch_metadata"] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "certipy_input": str(certipy_path),
        "target_domain": domain,
        "ca_nodes_added": len(ca_nodes),
        "template_nodes_added": len(template_nodes),
        "synthetic_edges_added": len(synthetic_edges),  # <-- Add this line
        "edges_injected": injected,
        "edges_skipped": skipped,
        "ntauth_link_established": domain_oid is not None
    }
    # -------------------------------

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if not args.quiet:
        kb = output_path.stat().st_size // 1024
        print(f"\nWritten {output_path} ({kb} KB)")
        print(f"  CA nodes        : {len(ca_nodes)}")
        print(f"  Template nodes  : {len(template_nodes)}")
        print(f"  Edges injected  : {injected}")
        print("\nNext: merger.py to combine this with other domains")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())