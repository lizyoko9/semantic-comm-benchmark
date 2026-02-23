"""Data loading factory."""

from data.cifar10 import get_cifar10_loaders
from data.kodak import get_kodak_loader


def get_dataloader(name: str, split: str = "train", batch_size: int = 128,
                   data_root: str = "./data_cache", **kwargs):
    """Create dataloader by dataset name.

    Returns (train_loader, test_loader) for train datasets,
    or single loader for eval-only datasets.
    """
    name = name.lower()
    if name == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(
            data_root=data_root, batch_size=batch_size,
            num_workers=kwargs.get("num_workers", 4),
        )
        return (train_loader, test_loader) if split == "both" else (
            train_loader if split == "train" else test_loader
        )
    elif name == "kodak":
        return get_kodak_loader(
            data_root=f"{data_root}/kodak", batch_size=batch_size,
            resize=kwargs.get("resize"),
        )
    else:
        raise ValueError(f"Unknown dataset: {name}. Available: cifar10, kodak")
