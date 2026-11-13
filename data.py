"""CIFAR-10 datasets and reproducible train/validation loaders."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def make_loaders(batch_size=128, seed=42, num_workers=0, download=False):
    """Return train, validation, and test loaders with a fixed 45k/5k split."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_data = datasets.CIFAR10(DATA_ROOT, train=True, transform=train_transform, download=download)
    validation_data = datasets.CIFAR10(DATA_ROOT, train=True, transform=eval_transform, download=False)
    test_data = datasets.CIFAR10(DATA_ROOT, train=False, transform=eval_transform, download=download)

    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_data), generator=split_generator).tolist()
    training = Subset(train_data, indices[5000:])
    validation = Subset(validation_data, indices[:5000])
    loader_generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=loader_generator),
        DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


if __name__ == "__main__":
    loaders = make_loaders(download=True)
    for name, loader in zip(("train", "validation", "test"), loaders):
        images, labels = next(iter(loader))
        print(f"{name}: {len(loader.dataset)} samples; images {tuple(images.shape)}; labels {tuple(labels.shape)}")
