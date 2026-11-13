"""Train a CIFAR-10 ResNet-18 or SE-ResNet-18 experiment."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data import make_loaders
from models import ResNet18


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("resnet18", "resnet18-se"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for this experiment")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0:
        parser.error("epochs, batch size and learning rate must be positive; workers must be nonnegative")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc="train" if training else "validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered")
            if training:
                loss.backward()
                optimizer.step()
            count += labels.size(0)
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
    return loss_sum / count, correct / count


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check your PyTorch and NVIDIA driver installation")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda")
    train_loader, validation_loader, _ = make_loaders(
        batch_size=args.batch_size, seed=args.seed, num_workers=args.workers
    )
    model = ResNet18(use_se=args.model == "resnet18-se").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    config = {
        "model": args.model, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "seed": args.seed, "num_workers": args.workers,
        "optimizer": "SGD", "momentum": 0.9, "weight_decay": 5e-4,
        "scheduler": "CosineAnnealingLR", "selection_metric": "validation_accuracy",
        "train_samples": len(train_loader.dataset), "validation_samples": len(validation_loader.dataset),
        "device": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["epoch", "learning_rate", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "seconds"]
    best_accuracy = -1.0
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_accuracy = run_epoch(model, validation_loader, criterion, device)
            row = {
                "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss, "train_accuracy": train_accuracy,
                "validation_loss": val_loss, "validation_accuracy": val_accuracy,
                "seconds": time.perf_counter() - start,
            }
            writer.writerow(row)
            file.flush()
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save({"model": args.model, "state_dict": model.state_dict(), "epoch": epoch,
                            "validation_accuracy": val_accuracy, "config": config}, args.output / "best.pt")
            scheduler.step()
            print(f"epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} train_acc={train_accuracy:.4%} "
                  f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4%} best_val_acc={best_accuracy:.4%}", flush=True)


if __name__ == "__main__":
    main()
