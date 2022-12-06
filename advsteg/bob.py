import torch
import torch.nn as nn

from advsteg.utils import conv_out


class BobDecoder(nn.Module):
    def __init__(self, msg_len: int = 100, output_size: int = 32) -> None:
        super(BobDecoder, self).__init__()
        h, w = output_size, output_size
        h_out, w_out = conv_out(h, 2**4), conv_out(w, 2**4)

        self.layers = nn.Sequential(
            # Layer 1: conv2d -> lrelu
            nn.Conv2d(3, output_size, 5, 2, 0),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2: conv2d * 2 -> bn -> lrelu
            nn.Conv2d(output_size, output_size * 2, 5, 2, 0),
            nn.BatchNorm2d(output_size * 2, 1e-5, 0.9),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3: conv2d * 4 -> bn -> lrelu
            nn.Conv2d(output_size * 2, output_size * 4, 5, 2, 0),
            nn.BatchNorm2d(output_size * 4, 1e-5, 0.9),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4: conv2d * 8 -> bn -> lrelu
            nn.Conv2d(output_size * 4, output_size * 8, 5, 2, 0),
            nn.BatchNorm2d(output_size * 8, 1e-5, 0.9),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.linear = nn.Linear(output_size * 8 * h_out * w_out, msg_len)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.layers(x)

        x = x.reshape([batch_size, -1])
        x = self.linear(x)
        x = self.tanh(x)
        return x
