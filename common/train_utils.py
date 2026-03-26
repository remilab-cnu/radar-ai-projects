"""Training loop utilities shared across all projects."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer,
                    device: str = "cpu") -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion,
             device: str = "cpu") -> float:
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def training_loop(model, train_loader, val_loader, criterion, optimizer,
                  epochs: int, checkpoint_dir: str | Path,
                  device: str = "cpu", scheduler=None) -> dict:
    """Standard training loop with best-val checkpoint saving."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    best_val = float("inf")

    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time"].append(dt)

        if scheduler is not None:
            scheduler.step(val_loss)

        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            improved = " *"

        if epoch <= 3 or epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}"
                  f"  {dt:.1f}s{improved}")

    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    total_time = sum(history["epoch_time"])
    print(f"  Done. Best val loss: {best_val:.4f}  Total time: {total_time:.0f}s")
    return history
