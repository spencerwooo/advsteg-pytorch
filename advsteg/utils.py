import math

import torch.nn as nn
from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def conv_out(size: int, stride: int) -> int:
    """This function is named conv_out_size_same in the original implementation."""
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


def track_metric(
    fields: list[str] = ["err"], format_str: str = ".4f", disable: bool = False
) -> Progress:
    """Create a progress bar using rich.progress along with a configurable field to
    update real-time metrics from the loop. Useful for training loops.

    ```python
    with track_metric(fields=[[FIELD]], disable=[DISABLE]) as tracker:
        t = tracker.add_task(description=[DESCRIPTION], total=[TOTAL], [FIELD]=0)
        for _ in range([TOTAL]):
            # ... loop here
            tracker.update(t, advance=1, [FIELD]=[VALUE])
    ```

    Args:
        fields: A list of metric names (fields) to track. Defaults to ["err"].
        disable: Render the progress bar or not. Defaults to False.

    Returns:
        The progress bar instance.
    """
    field_columns = [
        TextColumn(f"• {field} [yellow]{{task.fields[{field}]:{format_str}}}[yellow]")
        for field in fields
    ]

    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        "• ETA",
        TimeRemainingColumn(),
        *field_columns,
        disable=disable,
    )
