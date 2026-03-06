"""
client_malicious.py — Malicious (Byzantine) Flower Federated Learning Client
=============================================================================
Master's Thesis: "Layer-Wise Anomaly Detection for Byzantine-Robust
                  Aggregation in Federated Edge Learning"

Attack Type:  **Targeted Label Flipping**
    All images whose true label is "Dog" (class 5) are re-labelled to "Cat"
    (class 3) *before* training.  The model then learns, in good faith, on
    this poisoned dataset, producing weight updates that steer the global
    model toward confusing Dogs with Cats.

Why Dog → Cat?
    Dogs and Cats share significant visual features (fur, ears, eyes).
    This makes the poisoned updates less obviously anomalous than, say,
    flipping "Airplane" to "Frog" — the poisoned gradient direction is
    closer to the honest gradient, making the attack harder to detect.
    This is the standard "hard-case" label-flip benchmark in the Byzantine-
    robust FL literature.

Key Insight
-----------
The malicious client is structurally *identical* to the normal client —
it faithfully follows the Flower protocol and returns properly-formatted
weight updates.  The only modification is to the local dataset *before*
training begins.  This is what makes data-poisoning attacks so dangerous:
the server has no access to clients' raw data and cannot directly verify
its integrity.

This module is used to demonstrate that standard FedAvg (Phase 2) fails
under this attack, motivating the L-MAD defence mechanism.
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
# CIFAR-10 Class Index Reference (for clarity in comments)
# ──────────────────────────────────────────────────────────────────────────────
# 0: Airplane   1: Automobile   2: Bird    3: Cat     4: Deer
# 5: Dog        6: Frog         7: Horse   8: Ship    9: Truck
#
# Attack mapping:  Dog (5) ──→ Cat (3)

SOURCE_CLASS = 5   # Dog  — the class whose labels will be flipped
TARGET_CLASS = 3   # Cat  — the class it will be disguised as


# ──────────────────────────────────────────────────────────────────────────────
# Helper — apply_label_flip()
# ──────────────────────────────────────────────────────────────────────────────
def apply_label_flip(dataset, source: int, target: int) -> int:
    """
    Modify ``dataset.targets`` in-place, changing every occurrence of
    ``source`` to ``target``.

    Parameters
    ----------
    dataset : torchvision.datasets.CIFAR10
        The *underlying* CIFAR-10 dataset object.  When the dataset is
        wrapped in a ``Subset``, you must pass ``subset.dataset`` (the
        full dataset) rather than the ``Subset`` itself — see caller.
    source : int
        Original label index to be flipped (e.g. 5 = Dog).
    target : int
        New label index that replaces ``source`` (e.g. 3 = Cat).

    Returns
    -------
    num_flipped : int
        Number of labels that were changed (useful for logging).

    Notes
    -----
    The modification is performed *in-place* on the ``targets`` list.
    Because ``Subset`` holds a reference (not a copy) to the underlying
    dataset, any ``__getitem__`` call on the Subset will automatically
    see the modified labels.

    We intentionally only modify labels within the indices exposed by
    the client's Subset partition so that other clients sharing the same
    underlying dataset object are not accidentally poisoned.  (In the
    Flower simulation each client has its own process / data copy, but
    this makes the intent explicit.)
    """
    # dataset.targets is a Python list of ints for CIFAR-10
    targets = dataset.targets
    num_flipped = 0
    for i in range(len(targets)):
        if targets[i] == source:
            targets[i] = target
            num_flipped += 1
    return num_flipped


# ──────────────────────────────────────────────────────────────────────────────
# Malicious (Byzantine) Flower Client
# ──────────────────────────────────────────────────────────────────────────────
class FlowerMaliciousClient(fl.client.NumPyClient):
    """
    A Byzantine federated learning participant that performs a targeted
    Label Flipping attack: Dog (5) → Cat (3).

    Behaviour
    ---------
    1. Loads its CIFAR-10 partition (same IID split as the normal client).
    2. **Poisons the partition** by flipping all Dog labels to Cat.
    3. Trains *honestly* on the poisoned data — the model "learns" that
       dogs are cats, producing weight updates biased in that direction.
    4. Returns the biased weights to the server as if nothing happened.

    Everything else (``get_parameters``, ``set_parameters``, ``fit``,
    ``evaluate``) is structurally identical to ``FlowerNormalClient``.

    Parameters
    ----------
    partition_id : int
        Zero-based index identifying which data shard this client owns.
    num_partitions : int
        Total number of federated clients.
    """

    def __init__(self, partition_id: int, num_partitions: int) -> None:
        super().__init__()
        self.partition_id = partition_id
        self.num_partitions = num_partitions

        # ---- Instantiate the local model on the GPU/CPU ----
        self.model = SimpleCNN().to(DEVICE)

        # ---- Load this client's data partition ----
        self.trainloader, self.testloader = load_data(
            partition_id=self.partition_id,
            num_partitions=self.num_partitions,
        )

        # ================================================================
        # *** LABEL FLIPPING ATTACK ***
        # ================================================================
        # The trainloader wraps a Subset, which wraps the full CIFAR-10
        # dataset.  We flip labels on the *underlying* dataset object.
        #
        # Subset → dataset  : the actual CIFAR-10 Dataset
        # Subset → indices   : which samples belong to this client
        #
        # IMPORTANT: We only flip labels at indices belonging to *this*
        # client's partition.  This prevents cross-contamination when
        # multiple clients share the same in-memory dataset object (e.g.
        # in Flower's simulation mode).
        # ================================================================
        subset = self.trainloader.dataset  # this is a torch Subset
        underlying_dataset = subset.dataset
        partition_indices = subset.indices

        num_flipped = 0
        for idx in partition_indices:
            if underlying_dataset.targets[idx] == SOURCE_CLASS:
                underlying_dataset.targets[idx] = TARGET_CLASS
                num_flipped += 1

        self.num_train_samples = len(partition_indices)

        print(
            f"[MALICIOUS Client {self.partition_id}] Initialised — "
            f"{self.num_train_samples} training samples, "
            f"{num_flipped} labels flipped (Dog→Cat), device={DEVICE}"
        )

    # ------------------------------------------------------------------
    # get_parameters:  Model weights → list of NumPy arrays
    # ------------------------------------------------------------------
    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """
        Extract current model weights as NumPy arrays.

        Identical to the normal client — the attack is in the *data*, not
        in the weight serialisation.
        """
        return [
            val.cpu().numpy()
            for val in self.model.state_dict().values()
        ]

    # ------------------------------------------------------------------
    # set_parameters:  list of NumPy arrays → Model weights
    # ------------------------------------------------------------------
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Load NumPy arrays into the local PyTorch model.

        Identical to the normal client.
        """
        params_dict = zip(
            self.model.state_dict().keys(),
            parameters,
        )
        state_dict = OrderedDict(
            {name: torch.tensor(arr) for name, arr in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------
    # fit:  Local training round (on POISONED data)
    # ------------------------------------------------------------------
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Receive global weights, train on the *poisoned* partition, and
        return biased weight updates.

        The training procedure itself is completely standard — the bias
        comes solely from the corrupted labels, not from any modification
        to the optimisation process.  This makes the attack "stealthy":
        the magnitude and structure of the weight updates look normal to
        a naïve aggregator like FedAvg.
        """
        # Step 1 — Load the latest global weights
        self.set_parameters(parameters)

        # Step 2 — Train on the POISONED local data
        avg_loss = train(
            model=self.model,
            trainloader=self.trainloader,
            epochs=1,
            device=DEVICE,
        )
        print(
            f"[MALICIOUS Client {self.partition_id}] "
            f"Training loss (poisoned) = {avg_loss:.4f}"
        )

        # Step 3 — Return updated weights (biased toward Dog→Cat confusion)
        return (
            self.get_parameters(config={}),
            self.num_train_samples,
            {"train_loss": float(avg_loss)},
        )

    # ------------------------------------------------------------------
    # evaluate:  Local evaluation on the CLEAN test set
    # ------------------------------------------------------------------
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate on the *unmodified* CIFAR-10 test set.

        Even the malicious client evaluates honestly — the test set is
        not poisoned.  This lets the server track the *real* accuracy of
        the global model, including the damage caused by the attack.
        """
        self.set_parameters(parameters)

        loss, accuracy = test(
            model=self.model,
            testloader=self.testloader,
            device=DEVICE,
        )
        print(
            f"[MALICIOUS Client {self.partition_id}] "
            f"Test loss = {loss:.4f}, accuracy = {accuracy:.4f}"
        )

        return (
            float(loss),
            len(self.testloader.dataset),
            {"accuracy": float(accuracy)},
        )
