# L-MAD: Layer-Wise Median Absolute Deviation for Byzantine-Robust Aggregation in Federated Edge Learning

> **Master's Thesis** — Layer-Wise Anomaly Detection for Byzantine-Robust Aggregation in Federated Edge Learning

## Abstract

This repository contains the complete simulation codebase for the **FedLMAD** aggregation strategy, a novel defence mechanism against Byzantine poisoning attacks in Federated Learning. Unlike conventional whole-client rejection schemes (e.g., Krum, Trimmed Mean), L-MAD operates at **layer granularity**: each parameter tensor of every client update is independently scored against the population consensus using the Median Absolute Deviation (MAD), and only the anomalous layers are rejected. This preserves useful gradient information from partially-poisoned or statistically heterogeneous clients.

The simulation creates a 10-client federation — 7 honest and 3 malicious — over the CIFAR-10 dataset. Malicious clients execute a **targeted label-flipping attack** (Dog → Cat). The codebase supports head-to-head comparison between standard Federated Averaging (FedAvg, no defence) and the proposed FedLMAD strategy.

---

## System Requirements

| Requirement | Version / Details |
|---|---|
| **Python** | ≥ 3.9 |
| **PyTorch** | ≥ 2.0 (CUDA build recommended; CPU fallback is automatic) |
| **torchvision** | matching PyTorch version |
| **Flower** | `flwr[simulation]` ≥ 1.26 (bundles Ray for simulation mode) |
| **GPU** | NVIDIA GPU with CUDA ≥ 12.0 recommended (not strictly required) |
| **OS** | Windows / Linux / macOS |

> **Note:** The simulation allocates 10% of the GPU to each of the 10 virtual clients via Ray. A GPU with ≥ 4 GB VRAM is sufficient for CIFAR-10 with the SimpleCNN architecture.

---

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install PyTorch with CUDA support

Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select the command matching your CUDA version. Example for CUDA 12.4:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

For CPU-only (not recommended for performance):

```bash
pip install torch torchvision
```

### 3. Install Flower with simulation support

```bash
pip install "flwr[simulation]"
```

### 4. Verify installation

```bash
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import flwr; print('Flower', flwr.__version__)"
```

---

## Execution

CIFAR-10 is downloaded automatically on the first run to the `data/` directory.

### Run the proposed method — FedLMAD (default)

```bash
python main.py --strategy lmad
```

### Run the baseline — FedAvg (no defence)

```bash
python main.py --strategy fedavg
```

### Custom hyperparameters

```bash
# More aggressive filtering (lower τ) over 20 rounds
python main.py --strategy lmad --tau 2.5 --rounds 20

# More permissive filtering (higher τ)
python main.py --strategy lmad --tau 4.0 --rounds 10
```

### Command-line arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--strategy` | `str` | `lmad` | `"fedavg"` (baseline) or `"lmad"` (proposed defence) |
| `--rounds` | `int` | `10` | Number of federated learning rounds |
| `--tau` | `float` | `3.0` | L-MAD anomaly score threshold (only used with `--strategy lmad`) |

---

## Project Structure

```
L-MAD/
├── main.py                  # Simulation orchestrator: argument parsing, client
│                            # factory, centralized evaluation, results serialization
├── model.py                 # SimpleCNN architecture, CIFAR-10 data loading (IID
│                            # partitioning), train() and test() loops
├── client_normal.py         # Honest Flower client — trains on unmodified data
├── client_malicious.py      # Byzantine Flower client — executes Dog→Cat label
│                            # flipping before honest training
├── strategy_lmad.py         # FedLMAD aggregation strategy — layer-wise MAD
│                            # anomaly detection (core thesis contribution)
├── data/                    # Auto-downloaded CIFAR-10 dataset
│   └── cifar-10-batches-py/
├── results/                 # Output directory (created at runtime)
│   ├── fedavg_results.json  # Per-round metrics for FedAvg
│   └── lmad_results.json    # Per-round metrics for FedLMAD
├── README.md                # This file
├── ARCHITECTURE.md          # System design and L-MAD methodology
└── EXPERIMENTS.md           # Experimental setup and evaluation guide
```

### Module Summaries

| File | Purpose |
|---|---|
| `main.py` | Entry point. Creates the 10-client federation (3 malicious, 7 honest), configures either FedAvg or FedLMAD as the aggregation strategy, launches the Flower simulation via Ray, and saves per-round accuracy/loss to JSON. |
| `model.py` | Defines `SimpleCNN` — a two-block convolutional network for CIFAR-10 (Conv→ReLU→Pool × 2, then FC(4096→512→10)). Also provides `load_data()` for IID partitioning, and GPU-aware `train()` / `test()` loops. |
| `client_normal.py` | Implements `FlowerNormalClient(NumPyClient)`. Receives global weights, trains for 1 local epoch on its unmodified CIFAR-10 partition, and returns updated weights via the Flower protocol. |
| `client_malicious.py` | Implements `FlowerMaliciousClient(NumPyClient)`. Identical to the normal client except that, during initialisation, all Dog (class 5) labels in its partition are flipped to Cat (class 3) before training begins. |
| `strategy_lmad.py` | Implements `FedLMAD(FedAvg)`. Overrides `aggregate_fit()` to inject the layer-wise MAD filtering pipeline: per-layer median computation, L2 distance calculation, MAD-based anomaly scoring, and zero-trust gating with threshold τ. Accepted layers are aggregated via weighted average; fully-rejected layers fall back to the element-wise median. |

---

## Output Format

Each simulation run produces a JSON file in `results/` with the following structure:

```json
{
  "metadata": {
    "strategy": "lmad",
    "num_clients": 10,
    "num_malicious": 3,
    "malicious_ids": [0, 1, 2],
    "timestamp": "2026-03-06T14:30:00.000000",
    "device": "cuda"
  },
  "centralized_accuracy": [
    {"round": 0, "accuracy": 0.1000},
    {"round": 1, "accuracy": 0.4523},
    ...
  ],
  "centralized_loss": [
    {"round": 0, "loss": 2.3026},
    {"round": 1, "loss": 1.5431},
    ...
  ],
  "distributed_loss": [...],
  "distributed_accuracy": [...]
}
```

- **`centralized_accuracy`** — Global model accuracy on the full CIFAR-10 test set (10,000 images), evaluated on the server after each round. This is the primary metric for thesis comparison plots.
- **`centralized_loss`** — Corresponding CrossEntropy loss.
- **`distributed_*`** — Client-averaged metrics (when distributed evaluation is enabled).

---

## Citation

If you use this codebase in your research, please cite the associated thesis:

```
@mastersthesis{lmad2026,
  title   = {Layer-Wise Anomaly Detection for Byzantine-Robust Aggregation
             in Federated Edge Learning},
  author  = {[Author Name]},
  school  = {[University]},
  year    = {2026}
}
```

---

## License

This project is developed as part of a Master's thesis. Please contact the author for licensing information.
