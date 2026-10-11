import torch
import torchvision
import torchvision.transforms as transforms


def get_dataloader(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Download (if needed) FashionMNIST and return its training data loader."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
