#!/bin/bash
# =============================================================================
# pipeline.sh — GNN-AD-Navigator full pipeline (prep + inference)
# =============================================================================
#
# Usage:
#   ./pipeline.sh <input_dir> <output_dir>  [--start NODE --target NODE]
#                                            [--model PATH]
#                                            [--model-type gcn|hgt]
#                                            [--beam N] [--depth N]
#                                            [--skip-prep]
#
# Modes:
#   1. Data preparation only (default if --start/--target absent):
#        ./pipeline.sh ./input ./output
#
#   2. Prep + inference query:
#        ./pipeline.sh ./input ./output \
#            --start "wley" --target "domain admins@inlanefreight.local"
#
#   3. Inference only (skip prep, reuse existing output/):
#        ./pipeline.sh ./input ./output \
#            --start "wley" --target "domain admins" --skip-prep
#
# Defaults:
#   --model      models/GCN.pt
#   --model-type gcn
#   --beam       3
#   --depth      6
#
# Output:
#   <output_dir>/
#     cleaned/              filtered bloodhound files
#     forest_graph.json     merged + stitched graph
#     heterodata.pt         tensors
#     pipeline.log          full run log
#     inference_<timestamp>.log   query output if inference run
# =============================================================================

set -e

BANNER=$(cat <<'EOF'
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║   ███▄    █  ▄▄▄     ██▒   █▓ ██▓  ▄████   ▄▄▄     ▄▄▄█████▒ ▒█████   ██▀███          ║
║   ██ ▀█   █ ▒████▄  ▓██░   █▒▓██▒ ██▒ ▀█▒▒████▄   ▓  ██▒ ▓▒ ▒██▒  ██▒▓██ ▒ ██▒        ║
║   ██  ▀█ ██▒▒██  ▀█▄ ▓██  █▒░▒██▒▒██░▄▄▄░▒██  ▀█▄ ▒ ▓██░ ▒░ ▒██░  ██▒▓██ ░▄█ ▒        ║
║   ██▒  ▐▌██▒░██▄▄▄▄██ ▒██ █░░░██░░▓█  ██▓░██▄▄▄▄██░ ▓██▓ ░  ▒██   ██░▒██▀▀█▄          ║
║   ██░   ▓██░ ▓█   ▓██▒ ▒▀█░  ░██░░▒▓███▀▒ ▓█   ▓██▒ ▒██▒ ░  ░ ████▓▒░░██▓ ▒██▒        ║
║   ░     ▒░   ▒▒   ▓▒█░  ░ ░   ░░   ░▒   ▒ ▒▒   ▓▒█░ ▒ ░       ▒░▒░▒░ ░ ▒▓ ░▒▓░        ║
║                                                                                       ║
║                                GNN-AD-NAVIGATOR v1.0                                  ║
║                       Active Directory Attack Path Discovery                          ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
EOF
)
echo "$BANNER"

#BANNER=$(cat <<'EOF'
#██▄    ███  ▀██ ██▀    ▄█▄  █▀  ▄▀▀▀▀█▄  ▄▄▄▄▀   ████▄ █▄▄▄▄ 
#  █    █ █ █  █  █     █   █  █▀   ▄ ▄▀ █  █  ▀▀▀ █    █   █ █   ▄▀ 
#  █   █  █ █▄▄█  █▀▀▀ █    █  █     ▀█▀ █▄▄█     █    █   █  █▀▀▌  
#  █  █   █ █  █   █   █    █  █▄    ▄█  █  █    █     ▀████  █  
#  ███▀   █    █    █▀▀    █     ▀███▀      █   ▀                
#        ▀    █    ▀      ▀             █                  ▀ 
#            ▀                         ▀                       
#            GNN-AD-NAVIGATOR · Attack Path Discovery
#EOF
#)
#    echo "$BANNER"


## BANNER block commented out as requested
# BANNER=$(cat <<'EOF'
#:::.    :::.  :::.  :::      .::.:::  .,-:::::/   :::. ::::::::::::   ...    :::::::..   
#`;;;;,  `;;;  ;;`;; ';;,   ,;;;' ;;;,;;-'````'    ;;`;;;;;;;;;;''''.;;;;;;;. ;;;;``;;;;  
#  [[[[[. '[[ ,[[ '[[,\[[  .[[/   [[[[[[   [[[[[[/,[[ '[[,   [[    ,[[     \[[,[[[,/[[['  
#  $$$ "Y$c$$c$$$cc$$$cY$c.$$"    $$$"$$c.    "$$c$$$cc$$$c  $$    $$$,     $$$$$$$$$c    
#  888    Y88 888   888,Y88P      888 `Y8bo,,,o88o888   888, 88,   "888,_ _,88P888b "88bo,
#  MMM     YM YMM   ""`  MP       MMM   `'YMUP"YMMYMM   ""`  MMM     "YMMMMMP" MMMM   "W" 
#EOF
#)
#    echo "$BANNER"

INPUT_DIR="${1:-}"
OUTPUT_DIR="${2:-}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$INPUT_DIR" && $SKIP_PREP -eq 0 ]]; then
    echo "ERROR: input dir not found: $INPUT_DIR"; exit 1
fi

SCRIPT_DIR="$PROJECT_DIR/scripts"

shift 2

# default inference options
START=""
TARGET=""
MODEL=""
MODEL_TYPE="hgt"
BEAM_WIDTH=3
MAX_DEPTH=6
SKIP_PREP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start)      START="$2"; shift 2 ;;
        --target)     TARGET="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --model-type) MODEL_TYPE="$2"; shift 2 ;;
        --beam)       BEAM_WIDTH="$2"; shift 2 ;;
        --depth)      MAX_DEPTH="$2"; shift 2 ;;
        --skip-prep)  SKIP_PREP=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# auto-resolve model path from model-type if user didn't pass --model
if [[ -z "$MODEL" ]]; then
    if [[ "$MODEL_TYPE" == "gcn" ]]; then
        MODEL="$PROJECT_DIR/models/GCN.pt"
    else
        MODEL="$PROJECT_DIR/models/HGT.pt"
    fi
fi

# decide if we're running inference
DO_INFERENCE=0
if [[ -n "$START" && -n "$TARGET" ]]; then
    DO_INFERENCE=1
fi

if [[ ! -d "$INPUT_DIR" && $SKIP_PREP -eq 0 ]]; then
    echo "ERROR: input dir not found: $INPUT_DIR"; exit 1
fi

SCRIPT_DIR="$PROJECT_DIR/scripts"
mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/pipeline.log"
# --- OPEN THE GLOBAL LOGGING VALVE ---
# Redirects both stdout (1) and stderr (2) through tee to append (-a) to the log


exec > >(tee -a "$LOG") 2>&1
echo "[+] Pipeline initialized at $(date +'%H:%M:%S')"

if [[ $SKIP_PREP -eq 0 ]]; then
    mkdir -p "$OUTPUT_DIR/cleaned"
    > "$LOG"

    echo "[*] Mode: Data Preparation"
    echo "    Input: $INPUT_DIR | Output: $OUTPUT_DIR"

    # ── stage 0: classify files by content ─────────────────────────────────
    echo ""
    echo "    [0/5] Classifying files..."

    python - "$INPUT_DIR" "$OUTPUT_DIR" <<'PY'
import json, sys, shutil
from pathlib import Path

input_dir  = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
bh_staging = output_dir / "raw_bloodhound"
bh_staging.mkdir(parents=True, exist_ok=True)

bloodhound, certipy, ignored = [], [], []

for f in sorted(input_dir.glob("*.json")):
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        ignored.append((f, f"unreadable: {e}")); continue
    if isinstance(data, dict):
        if "Certificate Templates" in data or "Certificate Authorities" in data:
            certipy.append(f); continue
        if isinstance(data.get("data"), list):
            bloodhound.append(f)
            shutil.copy(f, bh_staging / f.name); continue
    ignored.append((f, "unknown format"))

print(f"  Bloodhound scans : {len(bloodhound)}")
for f in bloodhound[:5]: print(f"    {f.name}")
if len(bloodhound) > 5: print(f"    ... and {len(bloodhound)-5} more")

print(f"  Certipy scans    : {len(certipy)}")
for f in certipy: print(f"    {f.name}")

if ignored:
    print(f"  Ignored          : {len(ignored)}")
    for f, reason in ignored: print(f"    {f.name}  ({reason})")

(output_dir / "_certipy_list.txt").write_text(
    "\n".join(str(p) for p in certipy)
)

if not bloodhound:
    print("\nERROR: no bloodhound scans found"); sys.exit(2)
PY

    # ── stage 1: filter ─────────────────────────────────────────────────────
    echo ""
    echo "    [1/5] Filtering raw scans..."
    python "$SCRIPT_DIR/clean_bloodhound.py" \
        --input  "$OUTPUT_DIR/raw_bloodhound" \
        --output "$OUTPUT_DIR/cleaned" \
        >> "$LOG" 2>&1 || { echo "FAIL: filter"; exit 2; }

    # ── stage 2: merge ──────────────────────────────────────────────────────
    echo ""
    echo "    [2/5] Merging bloodhound data..."
    python "$SCRIPT_DIR/merger.py" \
        --input  "$OUTPUT_DIR/cleaned" \
        --output "$OUTPUT_DIR/forest_graph.json" \
        >> "$LOG" 2>&1 || { echo "FAIL: merge"; exit 2; }

    # ── stage 3: stitch ─────────────────────────────────────────────────────
    echo ""
    echo "[3/5] Stitching ADCS data..."
    CERTIPY_LIST="$OUTPUT_DIR/_certipy_list.txt"
    if [[ -s "$CERTIPY_LIST" ]]; then
        while IFS= read -r certipy_file || [[ -n "$certipy_file" ]]; do
            [[ -z "$certipy_file" ]] && continue
            # Strip hidden carriage returns (\r) just in case
            certipy_file=$(echo "$certipy_file" | tr -d '\r')

            domain=$(python - "$certipy_file" <<'PY'
import json, re, sys

try:
    filepath = sys.argv[1].strip()
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    
    # 1. Try CA Subject (Most accurate: extracting DC=essos, DC=local)
    cas = data.get("Certificate Authorities", {}) or {}
    for key, ca in cas.items():
        if isinstance(ca, dict):
            subject = ca.get("Certificate Subject", "")
            dcs = re.findall(r"DC=([^,]+)", subject, flags=re.IGNORECASE)
            if dcs:
                print(".".join(dcs).strip().lower())
                sys.exit(0)
            
            # Fallback to DNS Name
            dns = ca.get("DNS Name", "")
            if "." in dns:
                print(".".join(dns.split(".")[1:]).strip().lower())
                sys.exit(0)

    # 2. Try Templates Fallback
    tmpls = data.get("Certificate Templates", {}) or {}
    for key, tmpl in tmpls.items():
        if isinstance(tmpl, dict):
            dn = tmpl.get("Distinguished Name", "")
            dcs = re.findall(r"DC=([^,]+)", dn, flags=re.IGNORECASE)
            if dcs:
                print(".".join(dcs).strip().lower())
                sys.exit(0)

except Exception as e:
    # Force the error to stderr so it prints to the terminal/log
    print(f"PYTHON PARSE ERROR: {e}", file=sys.stderr)

sys.exit(1)
PY
)
            echo "$certipy_file"
            echo 'hello5'
            if [[ -z "$domain" ]]; then
                echo "  ⚠ $(basename "$certipy_file"): could not detect domain — skipping"
                continue
            fi
            echo "  → $(basename "$certipy_file")  [domain: $domain]"
            # ── Stage 3: ADCS Stitching ────────────────────────────────────────────────
            echo "Running ADCS Stitcher..."
            python "$SCRIPT_DIR/stitcher.py" \
                --certipy "$certipy_file" \
                --domain  "$domain" \
                --input   "$OUTPUT_DIR/forest_graph.json" \
                --output  "$OUTPUT_DIR/forest_graph.json" \

            # The global exec block handles the logging automatically now!
        done < "$CERTIPY_LIST"
    else
        echo "  (no Certipy scans found)"
    fi
    # ── stage 4: build tensors ─────────────────────────────────────────────
    echo ""
    echo "    [4/5] Building tensors..."
    python "$SCRIPT_DIR/build_dataset.py" \
        --graph "$OUTPUT_DIR/forest_graph.json" \
        --out   "$OUTPUT_DIR" \
        >> "$LOG" 2>&1 || { echo "FAIL: build_dataset"; exit 2; }

    # ── stage 5: validate ──────────────────────────────────────────────────
    echo ""
    echo "    [5/5] Validating..."
    python "$SCRIPT_DIR/validate_dataset.py" \
        --hetero "$OUTPUT_DIR/heterodata.pt" \
        --graph  "$OUTPUT_DIR/forest_graph.json" \
        >> "$LOG" 2>&1 || echo "          (validation warnings — see log)"

    echo ""
    echo "============================================================"
    echo "Preparation complete"
    echo "============================================================"
    echo "  Forest graph : $OUTPUT_DIR/forest_graph.json"
    echo "  Hetero data  : $OUTPUT_DIR/heterodata.pt"
    echo ""
fi

# ── stage 6: inference (optional) ──────────────────────────────────────────
if [[ $DO_INFERENCE -eq 1 ]]; then
    if [[ ! -f "$OUTPUT_DIR/heterodata.pt" ]]; then
        echo "[-] ERROR: heterodata.pt not found — run prep first"; exit 1
    fi
    if [[ ! -f "$MODEL" ]]; then
        echo "[-] ERROR: model checkpoint not found: $MODEL"; exit 1
    fi

    ## Ensure logs directory exists even in skip-prep mode
    mkdir -p "$OUTPUT_DIR/logs"
    
    # Route inference logs securely
    TS=$(date +%Y%m%d_%H%M%S)
    INF_LOG="$OUTPUT_DIR/logs/inference_${TS}.log"

    echo "[*] Launching Inference Engine"
    echo "    Route: $START -> $TARGET"
    echo "    Model: $(basename "$MODEL") (Beam: $BEAM_WIDTH, Depth: $MAX_DEPTH)"
    
    # run inference, tee to both stdout and the inference log
    python "$SCRIPT_DIR/inference.py" \
        --hetero     "$OUTPUT_DIR/heterodata.pt" \
        --graph      "$OUTPUT_DIR/forest_graph.json" \
        --model      "$MODEL" \
        --model-type "$MODEL_TYPE" \
        --start      "$START" \
        --target     "$TARGET" \
        --beam-width "$BEAM_WIDTH" \
        --max-depth  "$MAX_DEPTH" \
        2>&1 | tee "$INF_LOG"
        
    echo "[✓] Log saved to: $INF_LOG"
fi