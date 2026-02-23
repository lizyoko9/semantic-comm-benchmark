"""CIFAR-10 data loading."""

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_cifar10_loaders(data_root: str = "./data_cache",
                        batch_size: int = 128,
                        num_workers: int = 4):
    """Get CIFAR-10 train and test DataLoaders.

    Images are normalized to [0, 1] for reconstruction tasks.
    """
    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),  # [0, 1]
    ])
    transform_test = T.Compose([
        T.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
