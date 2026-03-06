# Experimental Design and Evaluation

> **Thesis Chapter 4** — Layer-Wise Anomaly Detection for Byzantine-Robust Aggregation in Federated Edge Learning

This document specifies the experimental setup, evaluation protocol, planned experiments, and guidance for interpreting results. All configurations match the implemented codebase and can be reproduced exactly using the commands provided.

---

## Table of Contents

1. [Experimental Setup](#1-experimental-setup)
2. [Federation Configuration](#2-federation-configuration)
3. [Attack Specification](#3-attack-specification)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Experiment Matrix](#5-experiment-matrix)
6. [Expected Results and Hypotheses](#6-expected-results-and-hypotheses)
7. [Hyperparameter Sensitivity Analysis](#7-hyperparameter-sensitivity-analysis)
8. [Results Interpretation Guide](#8-results-interpretation-guide)

---

## 1. Experimental Setup

### 1.1 Dataset

| Property | Value |
|---|---|
| Dataset | CIFAR-10 |
| Training samples | 50,000 |
| Test samples | 10,000 |
| Image dimensions | $32 \times 32 \times 3$ (RGB) |
| Number of classes | 10 |
| Normalisation | Per-channel: $\mu = (0.4914, 0.4822, 0.4465)$, $\sigma = (0.2470, 0.2435, 0.2616)$ |

CIFAR-10 is a standard benchmark for Byzantine-robust FL research. Its 10 visually distinct classes (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck) allow targeted attacks between semantically similar pairs (Dog ↔ Cat).

### 1.2 Model Architecture

| Component | Configuration |
|---|---|
| Architecture | SimpleCNN (2 convolution blocks + 2 fully-connected layers) |
| Block 1 | Conv2d(3→32, 3×3, pad=1) → ReLU → MaxPool(2×2) |
| Block 2 | Conv2d(32→64, 3×3, pad=1) → ReLU → MaxPool(2×2) |
| Classifier | FC(4096→512) → ReLU → Dropout(0.5) → FC(512→10) |
| Total parameters | 2,122,186 |
| Parameter tensors ($L$) | 8 (4 weight matrices + 4 bias vectors) |

### 1.3 Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimiser | SGD | Standard in FL literature for reproducibility |
| Learning rate | 0.01 | Conservative, prevents divergence under poisoning |
| Momentum | 0.9 | Standard SGD momentum |
| Loss function | CrossEntropyLoss | Standard for multi-class classification |
| Local epochs per round | 1 | Minimises client drift; standard baseline |
| Batch size | 32 | Standard for CIFAR-10 |
| Weight initialisation | PyTorch default (Kaiming uniform) | All clients start from the same random seed |

### 1.4 Compute Resources

| Resource | Configuration |
|---|---|
| Device | NVIDIA GPU with CUDA (auto-detected; CPU fallback available) |
| Per-client GPU allocation | 10% (via Ray: `num_gpus=0.1`) |
| Per-client CPU allocation | 1 core |
| Simultaneous clients | Up to 10 on a single GPU |

---

## 2. Federation Configuration

| Parameter | Value | Code Reference |
|---|---|---|
| Total clients ($K$) | 10 | `NUM_CLIENTS = 10` |
| Malicious clients ($M$) | 3 (IDs: 0, 1, 2) | `NUM_MALICIOUS = 3` |
| Honest clients | 7 (IDs: 3, 4, 5, 6, 7, 8, 9) | — |
| Byzantine ratio ($M/K$) | 30% | — |
| Participation rate | 100% (`fraction_fit=1.0`) | All clients train every round |
| Data partitioning | IID (equal non-overlapping shards) | 5,000 samples per client |
| Evaluation mode | Centralised only (`fraction_evaluate=0.0`) | Server evaluates on full test set |
| Default number of rounds | 10 | `--rounds 10` |

### Client Identity Map

| Client ID | Type | Data Partition | Behaviour |
|---|---|---|---|
| 0 | **Malicious** | Samples 0–4,999 | Dog→Cat label flip, then honest training |
| 1 | **Malicious** | Samples 5,000–9,999 | Dog→Cat label flip, then honest training |
| 2 | **Malicious** | Samples 10,000–14,999 | Dog→Cat label flip, then honest training |
| 3 | Honest | Samples 15,000–19,999 | Unmodified training |
| 4 | Honest | Samples 20,000–24,999 | Unmodified training |
| 5 | Honest | Samples 25,000–29,999 | Unmodified training |
| 6 | Honest | Samples 30,000–34,999 | Unmodified training |
| 7 | Honest | Samples 35,000–39,999 | Unmodified training |
| 8 | Honest | Samples 40,000–44,999 | Unmodified training |
| 9 | Honest | Samples 45,000–49,999 | Unmodified training |

---

## 3. Attack Specification

| Property | Value |
|---|---|
| Attack type | Targeted Label Flipping |
| Source class | Dog (CIFAR-10 index 5) |
| Target class | Cat (CIFAR-10 index 3) |
| Attack surface | Training labels only (images unchanged) |
| Poisoning scope | Each malicious client's own partition only |
| Approximate flipped labels per client | ~500 (10% of 5,000 — the Dog class frequency) |
| Total flipped labels across all attackers | ~1,500 |
| Attack timing | One-time, at client initialisation (before round 1) |
| Protocol compliance | Full — malicious clients follow Flower protocol correctly |

### Why Dog → Cat?

CIFAR-10 class frequency is approximately uniform (5,000 per class in the full training set, ~500 per class per partition). Dogs and Cats share significant visual features (fur, ears, snout), making the poisoned gradient direction geometrically close to the honest gradient. This is the standard "hard-case" attack in the Byzantine FL literature — a semantically distant flip (e.g., Airplane → Frog) would produce outlier updates that are trivially detected by any distance-based defence.

---

## 4. Evaluation Metrics

### 4.1 Primary Metric

| Metric | Description | Source |
|---|---|---|
| **Centralized Test Accuracy** | Global model accuracy on the full CIFAR-10 test set (10,000 images), evaluated after each aggregation round | `evaluate_fn` in `main.py` |

This is the single metric used in head-to-head comparison plots (FedAvg vs. FedLMAD). It measures the *real-world* impact of both the attack and the defence on classification performance.

### 4.2 Secondary Metrics

| Metric | Description | Source |
|---|---|---|
| Centralized Test Loss | CrossEntropy loss on the full test set per round | `evaluate_fn` |
| Total Rejections per Round | Number of (client, layer) pairs rejected by L-MAD | `lmad_total_rejections` in strategy metrics |
| Per-Layer Rejection Count | Number of clients rejected for each specific layer | `lmad_layer_{l}_rejections` |
| Per-Client Training Loss | Average training loss reported by each client | `train_loss` in fit metrics |

### 4.3 Attack-Specific Metrics (Derived)

These are computed post-hoc from the primary metrics:

| Metric | Formula | Interpretation |
|---|---|---|
| **Accuracy Drop** | $\Delta_{\text{acc}} = \text{Acc}_{\text{clean}} - \text{Acc}_{\text{attack}}$ | How much the attack degrades the model |
| **Defence Recovery** | $R = \text{Acc}_{\text{L-MAD}} - \text{Acc}_{\text{FedAvg}}$ | How much accuracy L-MAD recovers |
| **Recovery Rate** | $R\% = R \;/\; \Delta_{\text{acc}} \times 100$ | Percentage of attack damage mitigated |

---

## 5. Experiment Matrix

### 5.1 Core Experiments

| # | Experiment | Command | Strategy | Attack | Purpose |
|---|---|---|---|---|---|
| E1 | FedAvg under attack | `python main.py --strategy fedavg` | FedAvg | 3 malicious (Dog→Cat) | Demonstrate vulnerability of baseline |
| E2 | FedLMAD under attack ($\tau = 3.0$) | `python main.py --strategy lmad` | FedLMAD | 3 malicious (Dog→Cat) | Evaluate default defence |

### 5.2 Hyperparameter Sensitivity Experiments

| # | Experiment | Command | $\tau$ | Purpose |
|---|---|---|---|---|
| E3 | Aggressive filtering | `python main.py --strategy lmad --tau 2.0` | 2.0 | Test tighter anomaly gate |
| E4 | Moderate filtering | `python main.py --strategy lmad --tau 2.5` | 2.5 | Intermediate sensitivity |
| E5 | Permissive filtering | `python main.py --strategy lmad --tau 4.0` | 4.0 | Test looser anomaly gate |
| E6 | Very permissive filtering | `python main.py --strategy lmad --tau 5.0` | 5.0 | Near-FedAvg behaviour |

### 5.3 Extended Training Experiments

| # | Experiment | Command | Rounds | Purpose |
|---|---|---|---|---|
| E7 | FedAvg — long run | `python main.py --strategy fedavg --rounds 20` | 20 | Observe long-term poisoning effect |
| E8 | FedLMAD — long run | `python main.py --strategy lmad --rounds 20` | 20 | Verify sustained defence over many rounds |

### 5.4 Recommended Execution Order

```bash
# Step 1: Baseline under attack
python main.py --strategy fedavg --rounds 10

# Step 2: Proposed defence (default τ)
python main.py --strategy lmad --rounds 10

# Step 3: τ sensitivity sweep
python main.py --strategy lmad --tau 2.0 --rounds 10
python main.py --strategy lmad --tau 2.5 --rounds 10
python main.py --strategy lmad --tau 4.0 --rounds 10
python main.py --strategy lmad --tau 5.0 --rounds 10

# Step 4: Extended training
python main.py --strategy fedavg --rounds 20
python main.py --strategy lmad --rounds 20
```

> **Note:** Results are saved to `results/<strategy>_results.json`. Rename output files between runs if you want to preserve results from different $\tau$ values (e.g., `lmad_tau2.0_results.json`).

---

## 6. Expected Results and Hypotheses

### Hypothesis 1: FedAvg Degrades Under Attack

**Prediction:** FedAvg accuracy will be measurably lower than an undefended, unpoisoned baseline. The 30% Byzantine ratio allows malicious clients to inject sufficient bias into the weighted average, progressively steering the global model toward confusing Dogs with Cats.

**Expected observation:** Accuracy plateaus at a lower level compared to clean training, with the specific classes Dog and Cat showing the most degradation.

### Hypothesis 2: FedLMAD Recovers Accuracy

**Prediction:** FedLMAD with $\tau = 3.0$ will achieve significantly higher accuracy than FedAvg under the same attack. The layer-wise MAD filter should identify and reject the poisoned layers (primarily `fc2.weight` and `fc2.bias`, which encode class decision boundaries), while preserving honest layers.

**Expected observation:** Final accuracy is closer to clean (unpoisoned) performance. The rejection logs should show that malicious clients' classifier layers are consistently flagged.

### Hypothesis 3: τ Controls the Accuracy–Robustness Trade-off

| $\tau$ Value | Expected Behaviour |
|---|---|
| $\tau = 2.0$ (aggressive) | Most poisoned layers rejected, but may also reject honest layers that are naturally distant from median — potential over-filtering leading to slight accuracy loss |
| $\tau = 3.0$ (default) | Good balance — rejects poisoned layers while accepting most honest updates |
| $\tau = 4.0$ (permissive) | Fewer rejections overall — may let some poisoned layers through, slightly lower defence effectiveness |
| $\tau = 5.0$ (very permissive) | Behaviour approaches FedAvg — most layers accepted regardless of anomaly score |

### Hypothesis 4: Rejection Concentrates in Classifier Layers

**Prediction:** The per-layer rejection counts (`lmad_layer_{l}_rejections`) should be highest for layers 6 (`fc2.weight`) and 7 (`fc2.bias`), which directly encode the Dog → Cat class mapping. Earlier convolutional layers (layers 0–3) should show fewer or zero rejections, as label-flipping primarily affects the decision boundary, not the learned visual features.

---

## 7. Hyperparameter Sensitivity Analysis

### 7.1 Threshold τ (tau)

| Property | Details |
|---|---|
| Parameter | `--tau` (command-line argument) |
| Default | 3.0 |
| Range | $(0, \infty)$ — practically, values between 1.5 and 6.0 are meaningful |
| Effect | Controls the anomaly gate: $S_k^{(l)} > \tau \Rightarrow \text{reject}$ |
| Analogy | Equivalent to a "$\tau$-sigma" rule using MAD instead of standard deviation |

**Trade-off:**
- **Lower $\tau$** → more rejections → stronger defence but risk of over-filtering honest updates → potentially slower convergence
- **Higher $\tau$** → fewer rejections → weaker defence but preserves more honest gradient information → performance approaches FedAvg

**Suggested sweep:** $\tau \in \{2.0, 2.5, 3.0, 3.5, 4.0, 5.0\}$

### 7.2 Stability Constant ε (epsilon)

| Property | Details |
|---|---|
| Parameter | Code-level only (not exposed via CLI) |
| Default | $10^{-10}$ |
| Purpose | Prevents division-by-zero in $S_k^{(l)} = d_k^{(l)} / (\text{MAD}^{(l)} + \epsilon)$ when $\text{MAD}^{(l)} = 0$ (perfect consensus) |
| Sensitivity | Extremely low — only relevant when all clients have identical distances to the median. Any value in $[10^{-12}, 10^{-6}]$ produces equivalent behaviour |

### 7.3 Interaction Effects

The effective behaviour of the $\tau$ threshold depends on:

1. **Byzantine ratio ($M/K$):** Higher compromise ratios shift the distance distribution, making it harder for the MAD to distinguish honest from malicious updates. The current 30% ratio ($M = 3, K = 10$) is below the theoretical breakdown point of the median ($\lfloor K/2 \rfloor = 5$).

2. **Attack strength:** Label-flipping is a moderate-strength attack. A model-replacement or gradient-scaling attack would produce much larger L2 distances and be easier for L-MAD to detect, even at higher $\tau$.

3. **Data heterogeneity:** In non-IID settings, honest clients naturally produce more diverse updates. The $\tau$ threshold may need to be increased to avoid false rejections.

---

## 8. Results Interpretation Guide

### 8.1 Output File Structure

Each experiment produces a JSON file in the `results/` directory:

```
results/
├── fedavg_results.json    # From experiment E1
└── lmad_results.json      # From experiment E2
```

### 8.2 JSON Schema

```json
{
  "metadata": {
    "strategy": "lmad | fedavg",
    "num_clients": 10,
    "num_malicious": 3,
    "malicious_ids": [0, 1, 2],
    "timestamp": "ISO 8601 datetime",
    "device": "cuda | cpu"
  },
  "centralized_accuracy": [
    {"round": 0, "accuracy": 0.1000},
    {"round": 1, "accuracy": 0.4523}
  ],
  "centralized_loss": [
    {"round": 0, "loss": 2.3026},
    {"round": 1, "loss": 1.5431}
  ],
  "distributed_loss": [],
  "distributed_accuracy": []
}
```

### 8.3 Reading the Results

| Field | How to Interpret |
|---|---|
| `centralized_accuracy[0]` (round 0) | Pre-training random baseline (~10% for 10-class CIFAR-10) |
| `centralized_accuracy[-1]` | Final model accuracy — **the primary comparison metric** |
| `centralized_loss` | Should decrease over rounds; sustained high loss indicates persistent poisoning effect |
| `distributed_*` | Empty when `fraction_evaluate=0.0` (centralised evaluation only) |

### 8.4 Generating Comparison Plots

To produce the thesis comparison plot (FedAvg vs. FedLMAD accuracy over rounds), load both JSON files and plot `centralized_accuracy`:

```python
import json
import matplotlib.pyplot as plt

with open("results/fedavg_results.json") as f:
    fedavg = json.load(f)
with open("results/lmad_results.json") as f:
    lmad = json.load(f)

rounds_fa = [e["round"] for e in fedavg["centralized_accuracy"]]
acc_fa    = [e["accuracy"] for e in fedavg["centralized_accuracy"]]

rounds_lm = [e["round"] for e in lmad["centralized_accuracy"]]
acc_lm    = [e["accuracy"] for e in lmad["centralized_accuracy"]]

plt.figure(figsize=(8, 5))
plt.plot(rounds_fa, acc_fa, "o--", label="FedAvg (no defence)")
plt.plot(rounds_lm, acc_lm, "s-",  label="FedLMAD (τ=3.0)")
plt.xlabel("Federated Round")
plt.ylabel("Test Accuracy")
plt.title("FedAvg vs. FedLMAD Under Label-Flipping Attack (30% Byzantine)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/accuracy_comparison.png", dpi=150)
plt.show()
```

### 8.5 Console Output During Simulation

The L-MAD strategy produces detailed per-layer logging during each round. Key patterns to watch for:

```
--- Layer 6 (shape=(10, 512)) ---
  L2 distances: [0.0234, 0.0251, 0.0312, 0.0098, 0.0105, ...]
  Anomaly scores: [2.1, 2.3, 4.8, 0.9, 1.0, ...]
  [REJECTED] Client 2 | Score 4.8000 > tau=3.0 | Distance 0.031200
```

- **Look for:** Consistently high anomaly scores for clients 0, 1, 2 (the malicious ones) in the classifier layers (layers 6–7).
- **Red flag:** If honest clients (3–9) are frequently rejected, $\tau$ may be too low.
- **Good sign:** If only malicious clients are rejected, and only in specific layers, L-MAD is functioning as designed.

### 8.6 Statistical Significance

For thesis-quality results, consider:

1. **Multiple runs:** Run each experiment 3–5 times with different random seeds (requires modifying `get_initial_parameters()` to accept a seed parameter).
2. **Report mean ± std:** Average the per-round accuracy across runs and report confidence intervals.
3. **Per-class accuracy:** Compute a confusion matrix at the final round to show that L-MAD specifically preserves Dog-vs-Cat discrimination that FedAvg loses.
