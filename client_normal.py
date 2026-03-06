"""
client_normal.py — Standard (Honest) Flower Federated Learning Client
=====================================================================
Master's Thesis: "Layer-Wise Anomaly Detection for Byzantine-Robust
                  Aggregation in Federated Edge Learning"

This module implements a *normal* (non-malicious) Flower NumPyClient.
It serves as the honest baseline against which the malicious client's
behaviour will be compared.

Workflow per Federated Round
----------------------------
1. The server sends the current global model parameters to this client.
2. ``fit()`` loads those parameters into the local CNN, trains for 1 epoch
   on the client's private CIFAR-10 partition, and returns the updated
   weights as NumPy arrays.
3. ``evaluate()`` measures the model's accuracy on the full CIFAR-10 test
   set so the server can track global performance.

Flower's NumPyClient convenience class handles the serialisation between
PyTorch tensors and the NumPy arrays that travel over the wire.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch

import flwr as fl

# Local project imports — model.py must be on the Python path
from model import SimpleCNN, DEVICE, load_data, train, test


# ──────────────────────────────────────────────────────────────────────────────
# Normal (Honest) Flower Client
# ──────────────────────────────────────────────────────────────────────────────
class FlowerNormalClient(fl.client.NumPyClient):
    """
    An honest federated learning participant.

    This client:
      • Receives global model weights from the server.
      • Trains on its own *unmodified* CIFAR-10 data partition for 1 epoch.
      • Returns the updated weights for aggregation.

    No data manipulation is performed — this is the "ground truth" behaviour
    that the aggregation strategy can trust.

    Parameters
    ----------
    partition_id : int
        Zero-based index identifying which data shard this client owns.
    num_partitions : int
        Total number of federated clients (determines shard size).
    """

    def __init__(self, partition_id: int, num_partitions: int) -> None:
        super().__init__()
        self.partition_id = partition_id
        self.num_partitions = num_partitions

        # ---- Instantiate the local model on the GPU/CPU ----
        self.model = SimpleCNN().to(DEVICE)

        # ---- Load this client's private data partition ----
        # Each client receives a disjoint IID shard of the CIFAR-10 training
        # set.  The test set is shared (full 10 000 images) so that every
        # client evaluates on the same benchmark.
        self.trainloader, self.testloader = load_data(
            partition_id=self.partition_id,
            num_partitions=self.num_partitions,
        )

        # Record the number of training samples for weighted aggregation.
        # Flower uses this to compute a weighted average proportional to
        # each client's dataset size (important when partitions are unequal).
        self.num_train_samples = len(self.trainloader.dataset)

        print(
            f"[Normal Client {self.partition_id}] Initialised — "
            f"{self.num_train_samples} training samples, device={DEVICE}"
        )

    # ------------------------------------------------------------------
    # get_parameters:  Model weights → list of NumPy arrays
    # ------------------------------------------------------------------
    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """
        Extract the current model weights and return them as a list of
        NumPy arrays.

        Flower's communication protocol requires weights in NumPy format.
        PyTorch stores weights on the GPU, so we first move each tensor
        to CPU before converting with ``.numpy()``.

        Returns
        -------
        List[np.ndarray]
            One array per layer in the model's state_dict (same order every
            time because Python dicts are insertion-ordered since 3.7).
        """
        # state_dict().values() yields tensors in a deterministic order
        # matching the layer definitions in SimpleCNN.__init__().
        return [
            val.cpu().numpy()
            for val in self.model.state_dict().values()
        ]

    # ------------------------------------------------------------------
    # set_parameters:  list of NumPy arrays → Model weights
    # ------------------------------------------------------------------
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Load a list of NumPy arrays into the local PyTorch model.

        This is the inverse of ``get_parameters``.  We reconstruct an
        ``OrderedDict`` keyed by the layer names from the model's
        ``state_dict`` and load it in one shot with ``load_state_dict``.

        Parameters
        ----------
        parameters : List[np.ndarray]
            Weight arrays received from the server (one per layer).
        """
        # Build {layer_name: tensor} pairs
        params_dict = zip(
            self.model.state_dict().keys(),  # layer names
            parameters,                       # NumPy arrays from server
        )
        state_dict = OrderedDict(
            {name: torch.tensor(arr) for name, arr in params_dict}
        )
        # strict=True (default) ensures every key matches — catches
        # architecture mismatches early.
        self.model.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------
    # fit:  Local training round
    # ------------------------------------------------------------------
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Receive global parameters, train locally, return updated weights.

        Flower Protocol
        ---------------
        Returns a 3-tuple:
          1. Updated weight arrays (List[np.ndarray]).
          2. Number of training samples used (int) — used by the server for
             weighted averaging in FedAvg / L-MAD.
          3. A metrics dictionary (Dict) — optional; we include the training
             loss so the server can log it.

        Parameters
        ----------
        parameters : List[np.ndarray]
            Global model weights broadcast by the server at the start of
            this round.
        config : Dict[str, str]
            Server-side configuration (e.g., learning rate overrides).
            Not used in Phase 1 but kept for forward compatibility.
        """
        # Step 1 — Load the latest global weights into the local model
        self.set_parameters(parameters)

        # Step 2 — Train for 1 local epoch on this client's data partition
        avg_loss = train(
            model=self.model,
            trainloader=self.trainloader,
            epochs=1,
            device=DEVICE,
        )
        print(
            f"[Normal Client {self.partition_id}] "
            f"Training loss = {avg_loss:.4f}"
        )

        # Step 3 — Return updated weights + metadata
        return (
            self.get_parameters(config={}),  # updated NumPy weight arrays
            self.num_train_samples,           # number of examples used
            {"train_loss": float(avg_loss)},  # metrics for server logging
        )

    # ------------------------------------------------------------------
    # evaluate:  Local evaluation on the shared test set
    # ------------------------------------------------------------------
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the global model on the full CIFAR-10 test set.

        Flower Protocol
        ---------------
        Returns a 3-tuple:
          1. Test loss (float).
          2. Number of test samples (int).
          3. Metrics dictionary — we include accuracy so it can be
             aggregated / logged centrally by the server.

        Parameters
        ----------
        parameters : List[np.ndarray]
            Global model weights to evaluate.
        config : Dict[str, str]
            Server-side evaluation config (unused in Phase 1).
        """
        # Load the global weights (we don't train here — just evaluate)
        self.set_parameters(parameters)

        # Run inference on the full test set
        loss, accuracy = test(
            model=self.model,
            testloader=self.testloader,
            device=DEVICE,
        )
        print(
            f"[Normal Client {self.partition_id}] "
            f"Test loss = {loss:.4f}, accuracy = {accuracy:.4f}"
        )

        return (
            float(loss),                       # scalar test loss
            len(self.testloader.dataset),       # 10 000 test samples
            {"accuracy": float(accuracy)},      # extra metric
        )
