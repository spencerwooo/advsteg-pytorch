from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from rich import print

from advsteg.alice import AliceEncoder
from advsteg.bob import BobDecoder
from advsteg.eve import EveSteganalyzer
from advsteg.utils import load_celeba, track_metric, weights_init


@click.command()
@click.option("--dataset-path", type=str, default="data/celeba/", help="Path to CelebA")
@click.option("--image-size", default=109, help="Image size")
@click.option("--output-size", default=64, help="Output size")
@click.option("--msg-length", default=100, help="Message length")
@click.option("--batch-size", default=12, help="Batch size")
@click.option("--num-workers", default=8, help="Number of dataloader workers")
@click.option("--fraction", default=1.0, help="Fraction of dataset to use")
@click.option("--lr", default=1e-4, help="Learning rate")
@click.option("--epochs", default=100, help="Number of epochs")
@click.option("--a", type=float, default=0.1, help="Eve loss weight")
@click.option("--b", type=float, default=0.3, help="Alice reconstruction loss weight")
@click.option("--c", type=float, default=0.6, help="Bob message loss weight")
@click.option("--save-path", default="runs/", help="Path to save runs")
@click.option("--cuda", is_flag=True, help="Use CUDA")
def main(
    dataset_path,
    image_size,
    output_size,
    msg_length,
    batch_size,
    num_workers,
    fraction,
    lr,
    epochs,
    a,
    b,
    c,
    save_path,
    cuda,
):
    wandb.config = {
        "epochs": epochs,
        "lr": lr,
        "a": a,
        "b": b,
        "c": c,
        "fraction": fraction,
        "image_size": image_size,
        "output_size": output_size,
        "msg_length": msg_length,
    }
    wandb.init(project="advsteg", config=wandb.config)

    save_path = Path(save_path) / wandb.run.name  # type: ignore
    save_path.mkdir(parents=True, exist_ok=True)
    print(
        "[bold green]advsteg[/bold green]: Saving runs to "
        f"[underline]{save_path}[/underline]"
    )

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    print("[bold green]advsteg[/bold green]: Using device:", str(device).upper())

    print(
        "[bold blue]advsteg[/bold blue]: Loading dataset from "
        f"[underline]{dataset_path}[/underline]"
    )
    dataloader = load_celeba(
        dataset_path, batch_size, image_size, num_workers, fraction
    )

    print("[bold blue]advsteg[/bold blue]: Initializing models and optimizers")
    anet = AliceEncoder(msg_length, image_size, output_size).to(device)
    bnet = BobDecoder(msg_length, output_size).to(device)
    enet = EveSteganalyzer(output_size).to(device)
    for model in [anet, bnet, enet]:
        model.apply(weights_init)

    abnet_optim = optim.Adam(list(anet.parameters()) + list(bnet.parameters()), lr=lr)
    evnet_optim = optim.SGD(enet.parameters(), lr=lr)

    criterion = nn.BCEWithLogitsLoss()

    # Start training
    print(f"[bold yellow]advsteg[/bold yellow]: Start training for {epochs} epochs")

    for epoch in range(epochs):
        with track_metric(
            fields=["abnet", "b_l2", "enet"],
            format_str=".4f",
        ) as tracker:
            task = tracker.add_task(
                f"Epoch {epoch}/{epochs}",
                total=len(dataloader),
                abnet=0.0,
                b_l2=0.0,
                enet=0.0,
            )

            for i, (images, _) in enumerate(dataloader):
                images = images.to(device)
                messages = torch.randint(0, 2, size=(images.shape[0], msg_length))
                messages = (messages * 2 - 1) / 2
                messages = messages.to(device)

                # Generate stego images with Alice
                stego_images = anet(images, messages)

                # Decode messages with Bob
                decoded_messages = bnet(stego_images)

                # Eve's targets for real images should be ones,
                eve_ones_targets = torch.ones(images.shape[0], 1).to(device)
                # ... and for steganographic images, zeros
                eve_zeros_targets = torch.zeros(images.shape[0], 1).to(device)

                # Optimizing for Eve's loss
                evnet_optim.zero_grad()

                # Passing real images to Eve with labels 1
                evnet_loss_r = criterion(enet(images), eve_ones_targets)
                evnet_loss_r.backward()

                # Passing stego images to Eve with labels 0
                evnet_loss_f = criterion(enet(stego_images.detach()), eve_zeros_targets)
                evnet_loss_f.backward()

                # Eve's loss is a combination of real and stego images
                evnet_loss = evnet_loss_r + evnet_loss_f
                evnet_optim.step()

                # Optimizing for Alice and Bob's loss
                abnet_optim.zero_grad()
                # Bob's loss is the l2 distance between original and decoded messages
                bnet_loss = (messages - decoded_messages).pow(2).mean()
                # Alice's loss between encoded stego images and original ones, together
                # with Bob's l2 distance loss, and Eve's loss (where the label is
                # flipped when training the generator) is jointly optimized
                abnet_loss = (
                    a * criterion(enet(stego_images), eve_ones_targets)
                    + b * (images - stego_images).abs().mean()
                    + c * bnet_loss
                )
                abnet_loss.backward()
                abnet_optim.step()

                tracker.update(
                    task,
                    advance=1,
                    abnet=abnet_loss.item(),
                    b_l2=bnet_loss.item(),
                    enet=evnet_loss.item(),
                )

                # Log losses through wandb
                if i % 10 == 0:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "abnet": abnet_loss.item(),
                            "b_l2": bnet_loss.item(),
                            "enet": evnet_loss.item(),
                        }
                    )

                # Log generated images
                if i % 1000 == 0:
                    # Log first 5 images
                    wandb.log(
                        {
                            "cover image": wandb.Image(images[:5]),
                            "stego image": wandb.Image(stego_images[:5]),
                        }
                    )

    # Save models
    print(
        "[bold green]advsteg[/bold green]: Saving models to "
        f"[underline]{save_path}[/underline]"
    )
    weights = {
        "anet": anet.state_dict(),
        "bnet": bnet.state_dict(),
        # "enet": enet.state_dict(),
    }
    torch.save(weights, save_path / "advsteg_weights.pt")


if __name__ == "__main__":
    main()
