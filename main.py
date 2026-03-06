"""
main.py — Federated Learning Simulation Runner
===============================================
Master's Thesis: "Layer-Wise Anomaly Detection for Byzantine-Robust
                  Aggregation in Federated Edge Learning"

This script orchestrates the complete FL simulation using Flower's
``start_simulation`` API.  It creates a federation of 10 clients —
7 honest and 3 malicious (label-flipping Dog→Cat) — and runs training
for 10 rounds under either:

  • **FedAvg**  — the baseline aggregation (no defence).
  • **FedLMAD** — the proposed Layer-wise MAD defence (thesis contribution).

Usage
-----
    # Run with L-MAD defence (default)
    python main.py --strategy lmad

    # Run with baseline FedAvg (no defence)
    python main.py --strategy fedavg

    # Custom threshold and rounds
    python main.py --strategy lmad --tau 2.5 --rounds 20

Prerequisites
-------------
    pip install flwr[simulation]     # includes Ray for simulation
    pip install torch torchvision    # CUDA build recommended

Output
------
Results are saved to ``results/<strategy>_results.json`` containing
per-round accuracy and loss for thesis plotting.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import flwr as fl
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg

# Local project imports
from model import SimpleCNN, DEVICE, load_data, test
from client_normal import FlowerNormalClient
from client_malicious import FlowerMaliciousClient
from strategy_lmad import FedLMAD

# ──────────────────────────────────────────────────────────────────────────────
# Simulation Configuration
# ──────────────────────────────────────────────────────────────────────────────
NUM_CLIENTS = 10        # Total number of federated clients
NUM_MALICIOUS = 3       # Number of malicious (label-flipping) clients

# Malicious clients are assigned the FIRST partition IDs.
# Client IDs 0, 1, 2 → Malicious (Dog→Cat label flip)
# Client IDs 3, 4, 5, 6, 7, 8, 9 → Normal (honest)
MALICIOUS_IDS = set(range(NUM_MALICIOUS))


# ──────────────────────────────────────────────────────────────────────────────
# Client Factory — client_fn
# ──────────────────────────────────────────────────────────────────────────────
# Flower's start_simulation calls this function every time it needs to
# create a client.  The ``cid`` parameter is a string representation of
# the client's partition ID (0 through NUM_CLIENTS-1).
#
# Flower 1.26.1 auto-detects the old-style ``cid: str`` signature and
# wraps it in a Context adaptor internally, so this simple signature works
# with the legacy start_simulation API.  The adaptor maps:
#   cid = context.node_config["partition-id"]  (values "0" through "9")
# ──────────────────────────────────────────────────────────────────────────────
def client_fn(cid: str) -> fl.client.Client:
    """
    Create and return a Flower Client for the given client ID.

    Parameters
    ----------
    cid : str
        Client partition ID as a string ("0" through "9").
        Assigned by Flower's simulation engine from
        ``context.node_config["partition-id"]``.

    Returns
    -------
    fl.client.Client
        A wrapped NumPyClient — either normal or malicious depending
        on whether this client's ID is in MALICIOUS_IDS.
    """
    partition_id = int(cid)

    if partition_id in MALICIOUS_IDS:
        # Clients 0, 1, 2 → Malicious (Dog→Cat label flip)
        print(f"[main] Creating MALICIOUS client for partition {partition_id}")
        client = FlowerMaliciousClient(
            partition_id=partition_id,
            num_partitions=NUM_CLIENTS,
        )
    else:
        # Clients 3–9 → Normal (honest)
        print(f"[main] Creating Normal client for partition {partition_id}")
        client = FlowerNormalClient(
            partition_id=partition_id,
            num_partitions=NUM_CLIENTS,
        )

    # NumPyClient must be converted to the base Client type for Flower's
    # simulation engine.  .to_client() handles the protocol translation.
    return client.to_client()


# ──────────────────────────────────────────────────────────────────────────────
# Centralized Evaluation Function
# ──────────────────────────────────────────────────────────────────────────────
# This function runs *on the server* after each aggregation round.
# It loads the aggregated global model and evaluates it on the FULL
# CIFAR-10 test set (10,000 images).  This gives us a single, consistent
# accuracy measurement per round — exactly what we need for the thesis
# comparison plots (FedAvg vs. FedLMAD under attack).
# ──────────────────────────────────────────────────────────────────────────────
def get_evaluate_fn():
    """
    Return a centralized evaluation function for the Flower server.

    The returned function is passed to the strategy's ``evaluate_fn``
    parameter.  Flower calls it after every aggregation round with
    the global model parameters (as NumPy arrays).

    Returns
    -------
    evaluate : Callable
        Function with signature:
        ``(server_round, parameters, config) → (loss, metrics_dict)``
    """
    # Load the full CIFAR-10 test set once (shared across all rounds).
    # We use partition_id=0 just to get the testloader — the test set is
    # always the complete 10,000 images regardless of partition.
    _, testloader = load_data(partition_id=0, num_partitions=NUM_CLIENTS)

    def evaluate(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate the global model on the full CIFAR-10 test set.

        Parameters
        ----------
        server_round : int
            Current round number (0 = initial model before any training).
        parameters : NDArrays
            List of NumPy arrays representing the global model weights.
        config : Dict[str, Scalar]
            Server-side configuration (unused).

        Returns
        -------
        (loss, {"accuracy": accuracy}) or None
        """
        # Instantiate a fresh model and load the global weights
        model = SimpleCNN().to(DEVICE)

        # Reconstruct the state_dict from NumPy arrays
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {name: torch.tensor(arr) for name, arr in params_dict}
        )
        model.load_state_dict(state_dict, strict=True)

        # Run evaluation on the full test set
        loss, accuracy = test(model, testloader, DEVICE)

        print(f"\n{'*'*70}")
        print(f"[Server Round {server_round}] CENTRALIZED EVALUATION")
        print(f"  Loss     = {loss:.4f}")
        print(f"  Accuracy = {accuracy:.4f}  ({accuracy*100:.2f}%)")
        print(f"{'*'*70}\n")

        return loss, {"accuracy": float(accuracy)}

    return evaluate


# ──────────────────────────────────────────────────────────────────────────────
# Initial Model Parameters
# ──────────────────────────────────────────────────────────────────────────────
def get_initial_parameters() -> fl.common.Parameters:
    """
    Create initial model parameters from a freshly-initialised SimpleCNN.

    By providing initial parameters to the strategy, we ensure that:
      (a) All clients start from the exact same random initialisation.
      (b) The centralized ``evaluate_fn`` can measure round-0 (pre-training)
          accuracy for the baseline reference.

    Returns
    -------
    fl.common.Parameters
        Serialised initial weights.
    """
    model = SimpleCNN()
    weights = [val.cpu().numpy() for val in model.state_dict().values()]
    return ndarrays_to_parameters(weights)


# ──────────────────────────────────────────────────────────────────────────────
# Results Saving
# ──────────────────────────────────────────────────────────────────────────────
def save_results(
    history: fl.server.history.History,
    strategy_name: str,
    output_dir: str = "results",
) -> str:
    """
    Save simulation metrics to a JSON file for thesis plotting.

    The output file contains:
      - Per-round centralized accuracy and loss
      - Per-round distributed (client-averaged) loss
      - Simulation metadata (strategy, clients, rounds, timestamp)

    Parameters
    ----------
    history : fl.server.history.History
        Flower History object returned by start_simulation.
    strategy_name : str
        "fedavg" or "lmad" — used in the filename.
    output_dir : str
        Directory to save results in (created if it doesn't exist).

    Returns
    -------
    filepath : str
        Path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Extract centralized evaluation metrics ----
    # history.metrics_centralized = {"accuracy": [(round, value), ...]}
    # history.losses_centralized  = [(round, loss), ...]
    centralized_accuracy = []
    centralized_loss = []

    if history.losses_centralized:
        for round_num, loss_val in history.losses_centralized:
            centralized_loss.append({
                "round": round_num,
                "loss": float(loss_val),
            })

    if "accuracy" in history.metrics_centralized:
        for round_num, acc_val in history.metrics_centralized["accuracy"]:
            centralized_accuracy.append({
                "round": round_num,
                "accuracy": float(acc_val),
            })

    # ---- Extract distributed evaluation metrics (client-averaged) ----
    distributed_loss = []
    if history.losses_distributed:
        for round_num, loss_val in history.losses_distributed:
            distributed_loss.append({
                "round": round_num,
                "loss": float(loss_val),
            })

    distributed_accuracy = []
    if "accuracy" in history.metrics_distributed:
        for round_num, acc_val in history.metrics_distributed["accuracy"]:
            distributed_accuracy.append({
                "round": round_num,
                "accuracy": float(acc_val),
            })

    # ---- Build the results dictionary ----
    results = {
        "metadata": {
            "strategy": strategy_name,
            "num_clients": NUM_CLIENTS,
            "num_malicious": NUM_MALICIOUS,
            "malicious_ids": sorted(MALICIOUS_IDS),
            "timestamp": datetime.now().isoformat(),
            "device": str(DEVICE),
        },
        "centralized_accuracy": centralized_accuracy,
        "centralized_loss": centralized_loss,
        "distributed_loss": distributed_loss,
        "distributed_accuracy": distributed_accuracy,
    }

    # ---- Write to JSON ----
    filepath = os.path.join(output_dir, f"{strategy_name}_results.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[main] Results saved to: {filepath}")
    return filepath


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """
    Parse arguments, configure the strategy, and run the FL simulation.
    """
    # ---- Command-line arguments ----
    parser = argparse.ArgumentParser(
        description=(
            "Federated Learning Simulation: FedAvg vs. FedLMAD "
            "under Label Flipping Attack"
        ),
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="lmad",
        choices=["fedavg", "lmad"],
        help=(
            "Aggregation strategy to use. "
            "'fedavg' = standard Federated Averaging (baseline, no defence). "
            "'lmad' = Layer-wise MAD anomaly detection (thesis contribution). "
            "Default: lmad"
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds. Default: 10",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=3.0,
        help=(
            "L-MAD anomaly score threshold (only used with --strategy lmad). "
            "Lower = more aggressive filtering. Default: 3.0"
        ),
    )
    args = parser.parse_args()

    # ---- Print simulation configuration ----
    print(f"\n{'#'*70}")
    print(f"#  FEDERATED LEARNING SIMULATION")
    print(f"#  Strategy     : {args.strategy.upper()}")
    print(f"#  Rounds       : {args.rounds}")
    print(f"#  Total Clients: {NUM_CLIENTS}")
    print(f"#  Malicious    : {NUM_MALICIOUS} (IDs: {sorted(MALICIOUS_IDS)})")
    print(f"#  Normal       : {NUM_CLIENTS - NUM_MALICIOUS}")
    print(f"#  Device       : {DEVICE}")
    if args.strategy == "lmad":
        print(f"#  L-MAD tau    : {args.tau}")
    print(f"{'#'*70}\n")

    # ---- Get initial model parameters ----
    initial_parameters = get_initial_parameters()

    # ---- Build the centralized evaluation function ----
    evaluate_fn = get_evaluate_fn()

    # ---- Configure the aggregation strategy ----
    # Shared strategy kwargs (same for both FedAvg and FedLMAD since
    # FedLMAD inherits from FedAvg and accepts all the same parameters).
    strategy_kwargs = dict(
        fraction_fit=1.0,               # Select ALL clients every round
        fraction_evaluate=0.0,          # Disable distributed evaluation —
                                        # we use centralized evaluate_fn instead
        min_fit_clients=NUM_CLIENTS,    # Wait for all 10 clients
        min_evaluate_clients=0,         # No distributed eval
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate_fn,        # Server-side evaluation each round
        initial_parameters=initial_parameters,
    )

    if args.strategy == "fedavg":
        # Baseline: standard Federated Averaging — no Byzantine defence.
        # Under the label-flipping attack, FedAvg will blindly aggregate
        # the poisoned weights, degrading model performance.
        strategy = FedAvg(**strategy_kwargs)
        print("[main] Using strategy: FedAvg (NO defence)\n")
    else:
        # Thesis contribution: L-MAD defence.
        # Filters poisoned layers before aggregation.
        strategy = FedLMAD(
            tau=args.tau,
            **strategy_kwargs,
        )
        print(f"[main] Using strategy: FedLMAD (tau={args.tau})\n")

    # ---- Run the Flower simulation ----
    # start_simulation uses Ray under the hood to create virtual clients.
    # Each client gets a fraction of the GPU so Ray can schedule them on
    # a single GPU concurrently.
    #
    # client_resources:
    #   num_cpus=1   — each client uses 1 CPU core
    #   num_gpus=0.1 — each client uses 10% of the GPU
    #                   (allows up to 10 clients on 1 GPU)
    #
    # NOTE: Requires ``pip install flwr[simulation]`` which installs Ray.
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.1},
    )

    # ---- Save results for thesis plotting ----
    filepath = save_results(
        history=history,
        strategy_name=args.strategy,
    )

    # ---- Print final summary ----
    print(f"\n{'#'*70}")
    print(f"#  SIMULATION COMPLETE")
    print(f"#  Strategy: {args.strategy.upper()}")
    print(f"#  Rounds  : {args.rounds}")
    print(f"#")

    # Print final accuracy from centralized evaluation
    if history.metrics_centralized.get("accuracy"):
        final_round, final_acc = history.metrics_centralized["accuracy"][-1]
        print(f"#  Final Accuracy (Round {final_round}): "
              f"{final_acc:.4f}  ({final_acc*100:.2f}%)")

    if history.losses_centralized:
        final_round, final_loss = history.losses_centralized[-1]
        print(f"#  Final Loss     (Round {final_round}): {final_loss:.4f}")

    print(f"#")
    print(f"#  Results saved to: {filepath}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
