"""Evaluate a saved model on the CIFAR-10 test split."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from data import make_loaders
from models import ResNet18


def count_macs(model, device):
    """Count Conv2d and Linear multiply-accumulates for one 32x32 image."""
    total = 0

    def hook(module, inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            total += output.numel() * kernel * (module.in_channels // module.groups)
        else:
            total += output.numel() * module.in_features

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.inference_mode():
        model(torch.zeros(1, 3, 32, 32, device=device))
    for handle in handles:
        handle.remove()
    return total


def latency_ms(model, device, warmup=50, repeats=300):
    sample = torch.zeros(1, 3, 32, 32, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        torch.cuda.synchronize()
        durations = []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(sample)
            end.record()
            end.synchronize()
            durations.append(start.elapsed_time(end))
    return {"mean_ms": float(np.mean(durations)), "median_ms": float(np.median(durations)),
            "warmup": warmup, "repeats": repeats, "batch_size": 1}


def save_plots(matrix, class_names, history_path, output):
    fig, ax = plt.subplots(figsize=(9, 8))
    picture = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(picture, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names, xlabel="Predicted", ylabel="True",
           title="CIFAR-10 test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center",
                    color="white" if matrix[row, column] > matrix.max() / 2 else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    with history_path.open(newline="", encoding="utf-8") as file:
        history = list(csv.DictReader(file))
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, title in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
        for split in ("train", "validation"):
            axis.plot(epochs, [float(row[f"{split}_{metric}"]) for row in history], label=split)
        axis.set(xlabel="Epoch", ylabel=title, title=title)
        axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Training run directory")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    output = args.run / "evaluation"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation directory is not empty: {output}")
    checkpoint = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    model_name = checkpoint["model"]
    if model_name not in ("resnet18", "resnet18-se"):
        raise ValueError(f"Unknown model: {model_name}")
    device = torch.device("cuda")
    model = ResNet18(use_se=model_name == "resnet18-se").to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    _, _, test_loader = make_loaders(batch_size=128, seed=checkpoint["config"]["seed"])
    class_names = test_loader.dataset.classes
    labels, predictions = [], []
    with torch.inference_mode():
        for images, targets in test_loader:
            predicted = model(images.to(device)).argmax(dim=1).cpu()
            labels.extend(targets.tolist())
            predictions.extend(predicted.tolist())
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    report = classification_report(labels, predictions, labels=list(range(len(class_names))),
                                   target_names=class_names, output_dict=True, zero_division=0)
    metrics = {
        "model": model_name, "checkpoint_epoch": checkpoint["epoch"],
        "validation_accuracy": checkpoint["validation_accuracy"],
        "test_samples": len(labels), "test_accuracy": report["accuracy"],
        "classification_report": report,
        "parameters": sum(p.numel() for p in model.parameters()),
        "conv_linear_macs_per_image": count_macs(model, device),
        "macs_scope": "Conv2d and Linear only; excludes batch norm, activations, pooling and SE elementwise multiplication",
        "checkpoint_bytes": (args.run / "best.pt").stat().st_size,
        "latency": latency_ms(model, device),
        "device": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted", *class_names])
        for name, row in zip(class_names, matrix.tolist()):
            writer.writerow([name, *row])
    save_plots(matrix, class_names, args.run / "history.csv", output)
    print(json.dumps({key: metrics[key] for key in (
        "model", "checkpoint_epoch", "validation_accuracy", "test_samples", "test_accuracy",
        "parameters", "conv_linear_macs_per_image", "latency")}, indent=2))


if __name__ == "__main__":
    main()
