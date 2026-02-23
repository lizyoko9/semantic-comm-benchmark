"""Kodak24 evaluation dataset."""

import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.utils import save_image


KODAK_URLS = [
    f"https://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png"
    for i in range(1, 25)
]


class KodakDataset(Dataset):
    """Kodak24 lossless image test set (768x512 or 512x768)."""

    def __init__(self, root: str = "./data_cache/kodak", download: bool = True,
                 resize: int = None):
        self.root = root
        os.makedirs(root, exist_ok=True)

        if download:
            self._download()

        self.paths = sorted([
            os.path.join(root, f) for f in os.listdir(root)
            if f.endswith(".png")
        ])

        transforms = [T.ToTensor()]
        if resize is not None:
            transforms.insert(0, T.Resize((resize, resize)))
        self.transform = T.Compose(transforms)

    def _download(self):
        """Download Kodak images if not present."""
        import urllib.request
        for i, url in enumerate(KODAK_URLS, 1):
            path = os.path.join(self.root, f"kodim{i:02d}.png")
            if not os.path.exists(path):
                try:
                    urllib.request.urlretrieve(url, path)
                except Exception as e:
                    print(f"Warning: Could not download {url}: {e}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def get_kodak_loader(data_root: str = "./data_cache/kodak",
                     batch_size: int = 1, resize: int = None):
    """Get Kodak24 DataLoader for evaluation."""
    dataset = KodakDataset(root=data_root, download=True, resize=resize)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
