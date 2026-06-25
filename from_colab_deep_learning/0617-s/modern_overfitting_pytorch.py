"""
Modern overfitting-prevention training loop for the 0617 Fashion-MNIST PyTorch lab.

This is a practical upgrade over the notebook baseline:
- compact CNN instead of MLP
- torchvision v2 augmentation plus optional MixUp/CutMix
- AdamW, label smoothing, ReduceLROnPlateau
- EarlyStopping with min_delta, warm-up, checkpoint, best-weight restore
- optional EMA weights for more stable validation/final evaluation
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision import datasets
from torchvision.transforms import v2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED = 42


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RegularizedCNN(nn.Module):
    """Small CNN that uses image structure better than the MLP lab model."""

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ModelEMA:
    """Lightweight EMA wrapper with a state_dict compatible eval model."""

    def __init__(self, model: nn.Module, decay: float = 0.995, device: torch.device | None = None):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        if device is not None:
            self.ema.to(device)
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.ema.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value.to(ema_value.device), alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value.to(ema_value.device))


@dataclass
class EarlyStopping:
    patience: int = 6
    min_delta: float = 1e-4
    mode: str = "min"
    start_from_epoch: int = 5
    checkpoint_path: str | Path | None = "best_fashion_cnn.pt"

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.best = float("inf") if self.mode == "min" else -float("inf")
        self.bad_epochs = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_epoch = 0

    def _improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float, model: nn.Module, epoch: int) -> bool:
        if epoch < self.start_from_epoch:
            return False

        if self._improved(value):
            self.best = value
            self.best_epoch = epoch
            self.bad_epochs = 0
            self.best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
            if self.checkpoint_path is not None:
                torch.save(
                    {"epoch": epoch, "monitor": value, "model_state": self.best_state},
                    self.checkpoint_path,
                )
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: nn.Module, device: torch.device = DEVICE) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            model.to(device)


def build_loaders(
    batch_size: int = 128,
    train_size: int = 5000,
    val_size: int = 5000,
    use_mix: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomErasing(p=0.15, scale=(0.02, 0.10)),
            v2.Normalize((0.5,), (0.5,)),
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5,), (0.5,)),
        ]
    )

    full_train_aug = datasets.FashionMNIST("./data", train=True, download=True, transform=train_transform)
    full_train_eval = datasets.FashionMNIST("./data", train=True, download=True, transform=eval_transform)
    test_dataset = datasets.FashionMNIST("./data", train=False, download=True, transform=eval_transform)

    generator = torch.Generator().manual_seed(SEED)
    all_indices = torch.randperm(len(full_train_aug), generator=generator).tolist()
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size : train_size + val_size]

    train_dataset = Subset(full_train_aug, train_indices)
    val_dataset = Subset(full_train_eval, val_indices)

    mixup_or_cutmix = v2.RandomChoice(
        [
            v2.MixUp(num_classes=NUM_CLASSES, alpha=0.2),
            v2.CutMix(num_classes=NUM_CLASSES, alpha=1.0),
        ]
    )

    def collate_fn(batch):
        images, labels = default_collate(batch)
        if use_mix:
            images, labels = mixup_or_cutmix(images, labels)
        return images, labels

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn if use_mix else None,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    targets = labels.argmax(dim=1) if labels.ndim == 2 else labels
    return (preds == targets).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None = None,
    ema: ModelEMA | None = None,
    grad_clip: float | None = 1.0,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if ema is not None:
                    ema.update(model)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_acc += batch_accuracy(logits.detach(), labels.detach()) * batch_size
            total_count += batch_size

    return total_loss / total_count, total_acc / total_count


def train_modern(epochs: int = 50, use_mix: bool = True, use_ema: bool = True):
    set_seed()
    train_loader, val_loader, test_loader = build_loaders(use_mix=use_mix)

    model = RegularizedCNN(dropout=0.25).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-5,
    )
    ema = ModelEMA(model, decay=0.995, device=DEVICE) if use_ema else None
    early_stopping = EarlyStopping(
        patience=6,
        min_delta=1e-4,
        mode="min",
        start_from_epoch=5,
        checkpoint_path="best_fashion_cnn.pt",
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer, ema=ema)
        eval_model = ema.ema if ema is not None else model
        val_loss, val_acc = run_epoch(eval_model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if early_stopping.step(val_loss, eval_model, epoch):
            print(f"Early stopping at epoch {epoch}. Best epoch: {early_stopping.best_epoch}")
            break

    final_model = ema.ema if ema is not None else model
    early_stopping.restore(final_model)
    test_loss, test_acc = run_epoch(final_model, test_loader, criterion)
    print(f"Final test | loss={test_loss:.4f} acc={test_acc:.4f}")
    return final_model, history


if __name__ == "__main__":
    train_modern()
