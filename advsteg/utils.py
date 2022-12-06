import math

import torch.nn as nn
from rich import print
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def conv_out(size: int, stride: int) -> int:
    return int(math.ceil(float(size) / float(stride)))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def load_celeba(
    path: str,
    batch_size: int,
    image_size: int,
    num_workers: int = 8,
    fraction: float = 1.0,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = ImageFolder(path, transform=transform)
    if fraction < 1.0:
        dataset = Subset(dataset, range(int(len(dataset) * fraction)))
        print(f"[bold red]advsteg[/bold red]: Using only {len(dataset)} samples")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return dataloader
