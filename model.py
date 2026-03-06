"""
model.py — Core Model & Training Utilities for Federated Learning Simulation
=============================================================================
Master's Thesis: "Layer-Wise Anomaly Detection for Byzantine-Robust
                  Aggregation in Federated Edge Learning"

This module defines:
    1. SimpleCNN        – A lightweight 4-layer CNN for CIFAR-10 classification.
    2. load_data()      – Loads and partitions CIFAR-10 across federated clients.
    3. train()          – Standard PyTorch training loop (GPU-aware).
    4. test()           – Evaluation loop returning loss and accuracy.

Design Notes:
    • A simple CNN is preferred over ResNet-18 for fast iteration during
      simulation.  The architecture can be swapped without changing the
      federated pipeline because Flower operates on raw NumPy weight arrays.
    • CIFAR-10 images are 32×32×3.  After two 2×2 max-pool layers the spatial
      dimensions shrink to 8×8, giving a flattened feature vector of 64×8×8 =
      4 096 elements feeding into the first fully-connected layer.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ──────────────────────────────────────────────────────────────────────────────
# Device Selection
# ──────────────────────────────────────────────────────────────────────────────
# Automatically select GPU if an NVIDIA CUDA-capable device is available.
# This constant is importable by other modules (clients, server) so that
# every component uses the same device without duplication.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Model Architecture — SimpleCNN
# ──────────────────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network for CIFAR-10 (10 classes).

    Architecture
    ------------
    Layer Block 1:  Conv2d(3→32, 3×3, pad=1) → ReLU → MaxPool(2×2)
    Layer Block 2:  Conv2d(32→64, 3×3, pad=1) → ReLU → MaxPool(2×2)
    Classifier:     Flatten → FC(4096→512) → ReLU → Dropout(0.5) → FC(512→10)

    Input :  (batch, 3, 32, 32)   — CIFAR-10 images
    Output:  (batch, 10)          — logits for 10 classes

    The padding=1 in each Conv2d preserves spatial dimensions before pooling.
    After two 2×2 max-pool operations, 32→16→8, so the feature map is
    64 channels × 8 × 8 = 4 096 features.
    """

    def __init__(self) -> None:
        super().__init__()

        # --- Convolutional Block 1 ---
        # Input channels: 3 (RGB), Output channels: 32
        # Kernel size: 3×3, Padding: 1 (preserves 32×32 spatial dims)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1
        )
        # Max-Pool 2×2 → spatial dims become 16×16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Convolutional Block 2 ---
        # Input channels: 32, Output channels: 64
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        # After second pool → spatial dims become 8×8

        # --- Fully-Connected (Classifier) Layers ---
        # Flatten: 64 channels × 8 × 8 = 4 096
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        # Dropout prevents co-adaptation of features (regularisation)
        self.dropout = nn.Dropout(0.5)
        # Final output: 10 logits (one per CIFAR-10 class)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: convolutions → flatten → FC → logits."""
        # Block 1: Conv → ReLU → Pool
        x = self.pool(F.relu(self.conv1(x)))
        # Block 2: Conv → ReLU → Pool
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten all dimensions except the batch dimension
        x = x.view(x.size(0), -1)  # (batch, 4096)
        # Classifier head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # raw logits — no softmax (CrossEntropyLoss does it)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Data Loading & Partitioning — load_data()
# ──────────────────────────────────────────────────────────────────────────────

# Standard CIFAR-10 normalisation values (per-channel mean and std computed
# over the entire training set).  Using canonical values ensures the data
# distribution matches what the model expects.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 32,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 and return a *partitioned* training DataLoader plus the
    full test DataLoader.

    Partitioning Strategy (IID)
    ---------------------------
    The 50 000 training images are split into ``num_partitions`` equal,
    non-overlapping shards by simple index slicing.  This gives every
    federated client an identically distributed (IID) subset of the data — a
    common baseline before introducing heterogeneity experiments.

    Parameters
    ----------
    partition_id : int
        Zero-based index identifying which shard this client receives.
    num_partitions : int
        Total number of federated clients (= total shards).
    batch_size : int, default 32
        Mini-batch size for both train and test loaders.
    data_dir : str, default "./data"
        Directory where CIFAR-10 will be downloaded / cached.

    Returns
    -------
    trainloader : DataLoader
        Training data for this client's partition only.
    testloader : DataLoader
        Full CIFAR-10 test set (10 000 images) — same for all clients so
        that evaluation is comparable.
    """
    # --- Transforms ---
    # ToTensor converts PIL images (H×W×C, [0,255]) to tensors (C×H×W, [0,1]).
    # Normalize shifts and scales each channel to approximately zero mean and
    # unit variance using the pre-computed CIFAR-10 statistics.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # --- Download / Load CIFAR-10 ---
    trainset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    # --- IID Partitioning ---
    # Total training samples: 50 000
    total_samples = len(trainset)
    # Size of each partition (integer division; remainder samples are dropped
    # to keep all partitions the same size for fairness).
    samples_per_partition = total_samples // num_partitions
    # Compute the start and end indices for this client's shard.
    start_idx = partition_id * samples_per_partition
    end_idx   = start_idx + samples_per_partition
    # torch.utils.data.Subset wraps the original dataset and only exposes the
    # indices in the given range — no data is copied.
    partition_indices = list(range(start_idx, end_idx))
    train_partition = Subset(trainset, partition_indices)

    # --- DataLoaders ---
    trainloader = DataLoader(
        train_partition, batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Training Loop — train()
# ──────────────────────────────────────────────────────────────────────────────
def train(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int = 1,
    device: torch.device = DEVICE,
) -> float:
    """
    Train ``model`` on ``trainloader`` for the given number of epochs.

    Optimiser : SGD with learning-rate 0.01 and momentum 0.9.
    Loss      : CrossEntropyLoss (combines LogSoftmax + NLLLoss).

    Parameters
    ----------
    model : nn.Module
        The neural network (already on ``device``).
    trainloader : DataLoader
        Client's local training data partition.
    epochs : int, default 1
        Number of local epochs per federated round.
    device : torch.device
        Target device (CPU or CUDA GPU).

    Returns
    -------
    avg_loss : float
        Average training loss over all batches across all epochs.
    """
    # --- Optimiser & Loss ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Switch model to training mode (enables dropout, batch-norm updates, etc.)
    model.train()

    total_loss = 0.0
    total_batches = 0

    for _epoch in range(epochs):
        for images, labels in trainloader:
            # Move data to the same device as the model (GPU if available)
            images = images.to(device)
            labels = labels.to(device)

            # --- Standard PyTorch training step ---
            optimizer.zero_grad()           # Clear previous gradients
            outputs = model(images)         # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()                 # Backward pass (compute gradients)
            optimizer.step()                # Update weights

            total_loss += loss.item()
            total_batches += 1

    # Return the mean loss for logging / monitoring purposes
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Evaluation Loop — test()
# ──────────────────────────────────────────────────────────────────────────────
def test(
    model: nn.Module,
    testloader: DataLoader,
    device: torch.device = DEVICE,
) -> Tuple[float, float]:
    """
    Evaluate ``model`` on the full CIFAR-10 test set.

    Parameters
    ----------
    model : nn.Module
        The neural network (already on ``device``).
    testloader : DataLoader
        The complete CIFAR-10 test set DataLoader.
    device : torch.device
        Target device.

    Returns
    -------
    avg_loss : float
        Average CrossEntropyLoss over the test set.
    accuracy : float
        Fraction of correctly classified test images (0.0–1.0).
    """
    criterion = nn.CrossEntropyLoss()

    # Switch model to evaluation mode (disables dropout, freezes batch-norm)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # torch.no_grad() disables gradient computation — saves memory and speeds
    # up inference since we don't need gradients during evaluation.
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)  # weighted by batch sz

            # Predicted class = index of the maximum logit
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy
