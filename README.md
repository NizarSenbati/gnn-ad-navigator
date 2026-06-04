# GNN-AD-Navigator

> Graph Neural Network for Active Directory attack path discovery.
> Train on one environment, deploy on any unseen Active Directory graph.

```
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
```

## What it does

Given BloodHound output from any Active Directory environment, the
trained policy network discovers attack paths between specified nodes
using learned structural patterns of privilege escalation — without
hand-crafted rules or per-environment retraining.

Unlike traditional BloodHound queries that rely on operator expertise
to formulate Cypher, this tool **suggests** likely paths and annotates
each step with the exploitation technique. It then tries to **audits**
its own output to flag unreliable suggestions.

Trained on a single synthetic lab (GOAD, 351 nodes), the model
generalises to a 4000-node enterprise (INLANEFREIGHT) and rediscovers
the documented attack paths.

## Befor install

note 
Python 3.10+
1 GB disk (CPU install) or 3 GB (CUDA install) -pytorch is heavy on disk space
Inference runs comfortably on 8 GB RAM
Tested on Ubuntu 24.04


## Quick start

```bash
# 1. clone + install
git clone https://github.com/NizarSenbati/gnn-ad-navigator.git
cd gnn-ad-navigator

either use the normal setup script with direct pip commands or the docker file
./setup.sh

# build
docker build -t gnn-ad-navigator .

# run
docker run -v $(pwd)/input:/app/input \
           -v $(pwd)/output:/app/output \
           -it gnn-ad-navigator \
           ./pipeline.sh ./input ./output --start "wley" --target "DA"

# 2. drop your BloodHound scan into ./input/
#    (any *.json files from bloodhound-python, plus optional Certipy output)

# 3. run pipeline + inference in one command
./pipeline.sh ./input ./output \
    --start  "wley" \
    --target "domain admins@inlanefreight.local"


test with existing data in the examples folder:
`./launch.sh ./examples/input ./examples/output --start "wley" --target "domain admins@inlanefreight.local" --model-type hgt`

```

## Features

- **End-to-end pipeline** — raw BloodHound JSON → tensors → attack paths
- **Multi-domain forest support** — merges arbitrary numbers of domains
- **ADCS-aware** — stitches Certipy output into the graph
- **Trust traversal** — extracts and uses cross-domain trust edges
- - **Output audit (experimental)** — heuristic flagging of obviously degenerate paths; not a substitute for operator review
- **Operator advisories** — explains terminal states and next-step commands
- **Two architectures** — GCN baseline + HGT (heterogeneous graph transformer)

## Pipeline

```mermaid
flowchart LR
    bh[BloodHound JSON] --> filter[minimal_filter.py]
    cp[Certipy JSON] --> stitch[stitching.py]
    filter --> merge[merger.py]
    stitch --> merge
    merge --> forest[forest_graph.json]
    forest --> build[build_dataset.py]
    build --> tensors[heterodata.pt]
    tensors --> validate[validate_dataset.py]
    tensors --> infer[run_inference.py]
    infer --> out[Attack Paths + Audit]
```

Stages are individually runnable; `pipeline.sh` orchestrates them.

## A note on the AUDIT block

The AUDIT block beneath each path is a first attempt at automated reliability
flagging — it catches obvious failure modes (degenerate single-step paths,
missing DCSync at terminal, target-domain mismatch) but it isn't a substitute
for operator judgement. Treat it as a coarse signal that something might be
off, not as a verdict on path validity. In particular, it can flag a valid
path as suspicious if the model's preferred terminal happens to be one step
short of textbook DCSync. Future versions will refine the heuristics; for
now, read the path itself and decide.

## Installation

Requires Python 3.10+ and ~500MB of disk.

```bash
./setup.sh
```

This creates a virtualenv at `./venv`, installs all dependencies,
and makes the pipeline scripts executable. Activate with:

```bash
source venv/bin/activate
```

For manual install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
chmod +x pipeline.sh
```

## Usage

### Data preparation only

```bash
./pipeline.sh ./input ./output
```

Produces `output/forest_graph.json` and `output/heterodata.pt`.

### Inference query

```bash
./pipeline.sh ./input ./output \
    --start  "wley" \
    --target "domain admins@inlanefreight.local"
```

Outputs the top-K attack paths with annotated techniques, audit flags,
and operator advisories.

### Inference only (data already prepared)

```bash
./pipeline.sh ./input ./output \
    --start  "wley" \
    --target "domain admins" \
    --skip-prep
```

### Model selection

```bash
# default: GCN (fast, robust)
./pipeline.sh ./input ./output --start X --target Y

# HGT (heterogeneous attention, higher expressivity)
./pipeline.sh ./input ./output --start X --target Y --model-type hgt
```

## Example output

```
Path 1:
  Score: 1.0000  Steps: 4
    [1] wley@inlanefreight.local
        → damundsen@inlanefreight.local
        edge   : forcechangepassword
        attack : Password reset
        action : Force-change target's password
    [2] damundsen@inlanefreight.local
        → help desk level 1@inlanefreight.local
        edge   : genericwrite
        attack : ACL abuse — targeted write
    [3] help desk level 1@inlanefreight.local
        → information technology@inlanefreight.local
        edge   : memberof
        attack : Group inheritance
    [4] information technology@inlanefreight.local
        → adunn@inlanefreight.local
        edge   : genericall
        attack : ACL abuse — full control

  AUDIT:
    ✓ Path terminates at a node with DCSync on the target domain.
      This is operational success.

  ┌─ TERMINAL STATE REACHED ────────────────────────────────
  │  'adunn@inlanefreight.local' holds DCSync on inlanefreight.local
  │  Next: secretsdump.py -just-dc <user>:<pw>@<DC>
  └──────────────────────────────────────────────────────────
```

## Repository structure

```
gnn-ad-navigator/
├── README.md                    this file
├── LICENSE                      MIT
├── requirements.txt             Python dependencies
├── setup.sh                     install script
├── pipeline.sh                  pipeline launcher
│
├── scripts/                     pipeline components
│   ├── minimal_filter.py        drops dead AD objects
│   ├── stitching.py             injects ADCS data from Certipy
│   ├── merger.py                combines per-domain scans
│   ├── build_dataset.py         forest_graph → PyG tensors
│   ├── validate_dataset.py      pre-training audit
│   ├── prepare_training_examples.py    expert traces → labels
│   └── run_inference.py         beam-search + path audit
│
├── models/                      pre-trained checkpoints
│   ├── GCN.pt
│   └── HGT.pt
│
├── examples/                    minimal demo data
│   └── README.md
│
└── notebooks/                   Kaggle training notebooks
    ├── train_gcn.ipynb
    └── train_hgt.ipynb
```

## Limitations and known issues

This is research code. Known limitations:

- **Collection completeness affects results.** bloodhound-python does not
  reliably capture foreign group memberships across forests. SharpHound C#
  is recommended for cross-forest scenarios.
- **Confidence scores are relative, not absolute.** The audit module
  provides honest reliability assessment; raw scores can be misleading
  for short or degenerate paths.
- **Training set is small** (33 labelled transitions). The model handles
  structural ACL patterns well; less common edge types (HasSession,
  AllowedToDelegate variants) generalise less reliably.

See `docs/limitations.md` for the full discussion.

## Retraining on your own environment

Training is GPU-bound. Use the provided Kaggle notebooks:

1. Prepare your training graph with `pipeline.sh ./your_scan ./your_output`
2. Create `zoom.csv` with expert traces (see `examples/zoom_template.csv`)
3. Run `prepare_training_examples.py` to produce labels
4. Upload heterodata.pt + training_examples.json to Kaggle
5. Run `notebooks/train_gcn.ipynb` and `notebooks/train_hgt.ipynb`
6. Download the resulting checkpoints to `models/`

## Citation

This tool is the artefact of a Master's thesis:

```bibtex
@mastersthesis{senbati2026gnnavigator,
  author = {Senbati, [Your Full Name]},
  title  = {Learning Offensive Navigation Policies on Heterogeneous
            Attack Graphs: A Graph Neural Network Approach to
            Sequential Privilege Escalation in Active Directory},
  school = {[Your University]},
  year   = {2026},
}
```

## License

MIT — see [LICENSE](LICENSE) for details.

## Acknowledgements

- [GOAD](https://github.com/Orange-Cyberdefense/GOAD) — training environment
- [BloodHound](https://github.com/SpecterOps/BloodHound) — graph format
- [Certipy](https://github.com/ly4k/Certipy) — ADCS enumeration
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) — GNN backbone

Built as part of an M2 internship at [Your Institution]. Defended [Date].