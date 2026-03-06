"""
strategy_lmad.py — L-MAD: Layer-Wise Median Absolute Deviation Strategy
========================================================================
Master's Thesis: "Layer-Wise Anomaly Detection for Byzantine-Robust
                  Aggregation in Federated Edge Learning"

This module implements the core thesis contribution: **FedLMAD**, a custom
Flower aggregation strategy that defends against targeted poisoning attacks
(e.g., label flipping) by evaluating and filtering neural network weight
updates on a **layer-by-layer** basis.

Key Innovation
--------------
Unlike whole-client rejection schemes (e.g., Krum, Trimmed Mean), L-MAD
applies a **fine-grained, per-layer anomaly gate**.  A client whose
convolutional layers are honest but whose final classifier layer is
poisoned will only have the classifier layer rejected — the benign
convolutional updates are still aggregated.  This preserves more useful
information from partially-poisoned or heterogeneous clients.

Mathematical Pipeline (per layer l, across K clients)
------------------------------------------------------
1. **Layer-wise Median**:
       w_tilde^(l) = Median(w_1^(l), ..., w_K^(l))

2. **L2 Distance** for each client k:
       d_k^(l) = || w_k^(l) - w_tilde^(l) ||_2

3. **Median of Distances**:
       d_tilde^(l) = Median(d_1^(l), ..., d_K^(l))

4. **MAD (Median Absolute Deviation)**:
       MAD^(l) = Median( |d_k^(l) - d_tilde^(l)| )  for k = 1..K

5. **Anomaly Score** (modified z-score):
       S_k^(l) = |d_k^(l) - d_tilde^(l)| / (MAD^(l) + epsilon)

6. **Zero-Trust Gate**: If S_k^(l) > tau, **reject** layer l from
   client k.  Otherwise, accept it for aggregation.

7. **Weighted Average**: Aggregate only the accepted layers, weighted by
   each client's num_examples.

Parameters
----------
tau : float, default 3.0
    Anomaly score threshold.  Layers with S_k^(l) > tau are rejected.
    Lower values = more aggressive filtering; higher = more permissive.
epsilon : float, default 1e-10
    Small constant added to MAD to avoid division-by-zero when all clients
    are identical (MAD = 0).

Flower Integration
------------------
FedLMAD inherits from flwr.server.strategy.FedAvg and **only overrides
aggregate_fit**.  All other methods (client selection, evaluation
aggregation, parameter initialisation) are inherited from FedAvg unchanged.
This makes FedLMAD a drop-in replacement for FedAvg in any Flower pipeline.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


# ──────────────────────────────────────────────────────────────────────────────
# L-MAD Strategy
# ──────────────────────────────────────────────────────────────────────────────
class FedLMAD(FedAvg):
    """
    Federated Learning strategy with Layer-wise Median Absolute Deviation
    (L-MAD) anomaly detection for Byzantine-robust aggregation.

    Inherits from FedAvg so that client selection, evaluation aggregation,
    and parameter initialisation are handled identically to standard
    Federated Averaging.  Only ``aggregate_fit`` is overridden to inject
    the L-MAD filtering logic.

    Parameters
    ----------
    tau : float, default 3.0
        Anomaly score threshold. Any layer l of client k with
        S_k^(l) > tau is rejected from aggregation.
    epsilon : float, default 1e-10
        Numerical stability constant for MAD division.
    **kwargs
        All keyword arguments accepted by FedAvg (e.g.,
        fraction_fit, min_fit_clients, evaluate_fn, etc.).
    """

    def __init__(
        self,
        *,
        tau: float = 3.0,
        epsilon: float = 1e-10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tau = tau
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # aggregate_fit — CORE L-MAD LOGIC
    # ------------------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client weight updates using Layer-wise MAD filtering.

        This method replaces FedAvg's simple weighted average with a
        robust pipeline that:
          1. Extracts NumPy weight arrays from each client's FitRes.
          2. For each layer, computes the median, L2 distances, MAD,
             and anomaly scores.
          3. Rejects individual layers that exceed the threshold tau.
          4. Aggregates only the surviving (accepted) layers via
             weighted average.

        Parameters
        ----------
        server_round : int
            Current federated learning round number (1-indexed).
        results : List[Tuple[ClientProxy, FitRes]]
            Successful client updates. Each FitRes contains serialised
            parameters and the number of training examples used.
        failures : List[...]
            Failed client updates (logged but not used).

        Returns
        -------
        parameters : Parameters | None
            Aggregated global model parameters, or None if aggregation
            is not possible (e.g., no clients reported).
        metrics : Dict[str, Scalar]
            Aggregation metrics including the number of rejected layers.
        """
        # ==============================================================
        # STEP 0: Early-exit checks (same as FedAvg)
        # ==============================================================
        if not results:
            print(f"[L-MAD Round {server_round}] No results received. "
                  f"Skipping aggregation.")
            return None, {}

        if failures:
            print(f"[L-MAD Round {server_round}] "
                  f"{len(failures)} client(s) failed.")

        # ==============================================================
        # STEP 1: Extract weights and metadata from each client
        # ==============================================================
        # Number of participating clients
        K = len(results)
        print(f"\n{'='*70}")
        print(f"[L-MAD Round {server_round}] Aggregating updates from "
              f"{K} clients")
        print(f"{'='*70}")

        # Convert each client's Parameters into a list of NumPy arrays.
        # client_weights[k] = [layer_0_ndarray, layer_1_ndarray, ...]
        # client_num_examples[k] = number of training samples on client k
        client_weights: List[List[np.ndarray]] = []
        client_num_examples: List[int] = []

        for client_proxy, fit_res in results:
            # parameters_to_ndarrays: Parameters -> List[np.ndarray]
            weights = parameters_to_ndarrays(fit_res.parameters)
            client_weights.append(weights)
            client_num_examples.append(fit_res.num_examples)

        # Number of layers in the model (e.g., conv1.weight, conv1.bias, ...)
        num_layers = len(client_weights[0])
        print(f"[L-MAD Round {server_round}] Model has {num_layers} "
              f"parameter tensors (layers)")
        print(f"[L-MAD Round {server_round}] Threshold tau = {self.tau}, "
              f"epsilon = {self.epsilon}")

        # ==============================================================
        # STEP 2: Layer-wise L-MAD anomaly detection & filtering
        # ==============================================================
        # For each layer, we maintain a boolean mask indicating which
        # clients are accepted (True) or rejected (False).
        #
        # accepted_mask[l][k] = True   =>  layer l of client k is used
        # accepted_mask[l][k] = False  =>  layer l of client k is rejected

        # Track statistics for logging / thesis metrics
        total_rejections = 0
        layer_rejection_counts: List[int] = []

        # accepted_mask[l] is a list of K booleans
        accepted_mask: List[List[bool]] = []

        for l in range(num_layers):
            layer_shape = client_weights[0][l].shape
            print(f"\n--- Layer {l} (shape={layer_shape}) ---")

            # ----------------------------------------------------------
            # 2a. Stack all clients' weights for this layer.
            #     stacked shape: (K, *layer_shape)
            # ----------------------------------------------------------
            stacked = np.array([
                client_weights[k][l] for k in range(K)
            ])  # shape: (K, *layer_shape)

            # ----------------------------------------------------------
            # 2b. Compute the element-wise median across clients.
            #     median_w[i,j,...] = Median(w_1[i,j,...], ..., w_K[i,j,...])
            #
            #     This is the "robust center" of the weight space for
            #     this layer — the median is resilient to outliers,
            #     unlike the mean used in FedAvg.
            # ----------------------------------------------------------
            median_w = np.median(stacked, axis=0)  # shape: layer_shape
            print(f"  Median weight norm: {np.linalg.norm(median_w):.6f}")

            # ----------------------------------------------------------
            # 2c. Compute L2 distance from each client to the median.
            #     d_k = || w_k^(l) - median_w^(l) ||_2
            #
            #     A large distance means the client's update for this
            #     layer is far from what most clients agreed on.
            # ----------------------------------------------------------
            distances = np.array([
                np.linalg.norm(client_weights[k][l] - median_w)
                for k in range(K)
            ])  # shape: (K,)
            print(f"  L2 distances: {np.round(distances, 6)}")

            # ----------------------------------------------------------
            # 2d. Compute the median of the distances.
            #     d_tilde = Median(d_1, ..., d_K)
            #
            #     This represents the "typical" distance a client is
            #     from the median weights — the expected deviation.
            # ----------------------------------------------------------
            median_distance = np.median(distances)
            print(f"  Median distance: {median_distance:.6f}")

            # ----------------------------------------------------------
            # 2e. Compute MAD (Median Absolute Deviation).
            #     MAD = Median( |d_k - d_tilde| )  for k = 1..K
            #
            #     MAD is a robust measure of the "spread" of the
            #     distances.  Unlike standard deviation, it is not
            #     inflated by a single extreme outlier.  A MAD near
            #     zero means all clients have very similar distances
            #     to the median (high consensus).
            # ----------------------------------------------------------
            abs_deviations = np.abs(distances - median_distance)
            mad = np.median(abs_deviations)
            print(f"  MAD: {mad:.6f}")

            # ----------------------------------------------------------
            # 2f. Compute anomaly score for each client.
            #     S_k = |d_k - d_tilde| / (MAD + epsilon)
            #
            #     This is the modified z-score: it measures how many
            #     MADs each client's distance deviates from the median
            #     distance.  A score > tau means the client's update is
            #     an outlier *relative to the spread of the population*.
            #
            #     Using raw d_k / MAD would cause all clients to be
            #     rejected whenever MAD ≈ 0 (tight cluster of distances),
            #     which happens naturally as training converges and all
            #     clients update in similar directions.  The deviation
            #     |d_k - d_tilde| stays small for honest clients even
            #     when MAD is small, so the score correctly remains low.
            #
            #     epsilon prevents division by zero when MAD = 0
            #     (perfect consensus among all clients).
            # ----------------------------------------------------------
            scores = abs_deviations / (mad + self.epsilon)
            print(f"  Anomaly scores: {np.round(scores, 4)}")

            # ----------------------------------------------------------
            # 2g. Zero-Trust Gate: reject layers exceeding threshold tau.
            #
            #     tau = 3.0 by default, meaning a client's distance must
            #     be more than 3x the typical spread to be flagged.
            #     This is analogous to a "3-sigma" rule but using the
            #     robust MAD instead of standard deviation.
            # ----------------------------------------------------------
            layer_accepted = []
            layer_rejections = 0

            for k in range(K):
                if scores[k] > self.tau:
                    # REJECT this layer for this client
                    layer_accepted.append(False)
                    layer_rejections += 1
                    print(f"  [REJECTED] Client {k} | "
                          f"Score {scores[k]:.4f} > tau={self.tau} | "
                          f"Distance {distances[k]:.6f}")
                else:
                    # ACCEPT this layer for this client
                    layer_accepted.append(True)
                    print(f"  [accepted] Client {k} | "
                          f"Score {scores[k]:.4f} <= tau={self.tau}")

            accepted_mask.append(layer_accepted)
            layer_rejection_counts.append(layer_rejections)
            total_rejections += layer_rejections

            print(f"  >> Layer {l} summary: "
                  f"{K - layer_rejections}/{K} clients accepted, "
                  f"{layer_rejections} rejected")

        # ==============================================================
        # STEP 3: Aggregate accepted layers via weighted average
        # ==============================================================
        print(f"\n{'='*70}")
        print(f"[L-MAD Round {server_round}] AGGREGATION PHASE")
        print(f"  Total layer-level rejections: {total_rejections} "
              f"(out of {K * num_layers} total)")
        print(f"{'='*70}")

        aggregated_weights: List[np.ndarray] = []

        for l in range(num_layers):
            # Collect the accepted clients for this layer
            accepted_indices = [
                k for k in range(K) if accepted_mask[l][k]
            ]

            if len(accepted_indices) == 0:
                # EDGE CASE: All clients rejected for this layer.
                # Fallback to the element-wise median (the most robust
                # estimate when we cannot trust any single client).
                # This ensures the aggregation always produces a valid
                # model even under extreme attack scenarios.
                print(f"  Layer {l}: ALL clients rejected — "
                      f"falling back to median")
                stacked = np.array([
                    client_weights[k][l] for k in range(K)
                ])
                aggregated_layer = np.median(stacked, axis=0)
            else:
                # Weighted average of accepted clients, weighted by
                # num_examples (the standard FedAvg weighting scheme).
                #
                # aggregated = Sum(n_k * w_k) / Sum(n_k)  for accepted k
                #
                # This preserves the FedAvg property that clients with
                # more data have proportionally more influence — but only
                # if they pass the anomaly gate.
                total_examples = sum(
                    client_num_examples[k] for k in accepted_indices
                )
                aggregated_layer = np.zeros_like(client_weights[0][l])

                for k in accepted_indices:
                    # Weight contribution proportional to dataset size
                    weight = client_num_examples[k] / total_examples
                    aggregated_layer += weight * client_weights[k][l]

                print(f"  Layer {l}: Aggregated from clients "
                      f"{accepted_indices} "
                      f"(total examples: {total_examples})")

            aggregated_weights.append(aggregated_layer)

        # ==============================================================
        # STEP 4: Convert back to Flower Parameters and return
        # ==============================================================
        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)

        # Build metrics dictionary for server-side logging.
        # These metrics can be accessed after the simulation to plot
        # rejection rates, per-layer filtering behaviour, etc.
        metrics_aggregated: Dict[str, Scalar] = {
            "lmad_total_rejections": total_rejections,
            "lmad_num_clients": K,
            "lmad_num_layers": num_layers,
        }

        # Per-layer rejection counts in metrics (for plotting)
        for l in range(num_layers):
            metrics_aggregated[f"lmad_layer_{l}_rejections"] = (
                layer_rejection_counts[l]
            )

        # Aggregate client-reported metrics (e.g., train_loss) if a
        # custom aggregation function was provided via the constructor
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (res.num_examples, res.metrics) for _, res in results
            ]
            client_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
            metrics_aggregated.update(client_metrics)

        print(f"\n[L-MAD Round {server_round}] Aggregation complete.\n")

        return parameters_aggregated, metrics_aggregated
