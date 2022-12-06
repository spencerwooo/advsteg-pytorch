import torch
import torch.nn as nn

from advsteg.utils import conv_out


class AliceEncoder(nn.Module):
    def __init__(
        self, msg_len: int = 100, image_size: int = 62, output_size: int = 32
    ) -> None:
        super(AliceEncoder, self).__init__()
        h, w = output_size, output_size
        h_out, w_out = conv_out(h, 2**4), conv_out(w, 2**4)

        self.h_out, self.w_out = h_out, w_out
        self.output_size = output_size

        # Layer 1: fc layer (defined outside) -> bn -> relu
        self.linear = nn.Linear(
            image_size * image_size * 3 + msg_len, output_size * 8 * h_out * w_out
        )
        self.bn1 = nn.BatchNorm2d(output_size * 8)
        self.relu1 = nn.ReLU(inplace=True)

        self.layers = nn.Sequential(
            # Layer 2: deconv2d -> bn -> relu
            nn.ConvTranspose2d(output_size * 8, output_size * 4, 5, 2, 0),
            nn.BatchNorm2d(output_size * 4),
            nn.ReLU(inplace=True),
            # Layer 3: deconv2d -> bn -> relu
            nn.ConvTranspose2d(output_size * 4, output_size * 2, 5, 2, 0),
            nn.BatchNorm2d(output_size * 2),
            nn.ReLU(inplace=True),
            # Layer 4: deconv2d -> bn -> relu
            nn.ConvTranspose2d(output_size * 2, output_size, 5, 2, 0),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True),
            # Layer 5: deconv2d -> tanh
            nn.ConvTranspose2d(output_size, 3, 5, 2, 0),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        # Flatten the image and concatenate with the payload message
        batch_size = x.shape[0]
        x = x.reshape([batch_size, -1])
        x = torch.concat([x, msg], dim=1)

        # First fully-connected layer:
        # - allows the secret message to be combined with any region of the cover image
        x = self.linear(x)
        x = x.reshape([-1, self.output_size * 8, self.h_out, self.w_out])
        x = self.relu1(self.bn1(x))

        # CNN layers
        x = self.layers(x)
        return x
