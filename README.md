# advsteg-pytorch ![Python 3.10](https://img.shields.io/badge/Python%203.10-297ca0?logo=python&logoColor=white) [![Support Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFCC33?logo=WeightsAndBiases&logoColor=black)](https://wandb.ai) [![License MIT](https://img.shields.io/github/license/spencerwooo/advsteg-pytorch)](./LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

This is my attempt at reproducing the NeurIPS 2017 paper: [Generating Steganographic Images via Adversarial Training](https://papers.nips.cc/paper/2017/hash/fe2d010308a6b3799a3d9c728ee74244-Abstract.html) in modern PyTorch. I have not tested all declared experimental results in the paper, but basic functionality should be all available here.

## Usage

PyTorch related dependencies are defined in `environment.yml`:

```bash
conda env create -f environment.yml -n advsteg
```

Then activate the environment:

```bash
conda activate advsteg
```

*[Mamba](https://github.com/mamba-org/mamba) (drop-in replacement for conda) is also supported and is what I used for training.*

Dev-dependencies are defined and installed with Poetry:

```bash
# Inside the conda-created environment
poetry install
```

[Weights & Biases](https://wandb.ai/) is used for logging and visualization. You can either create an account and login with `wandb login` or add the environment variable `WANDB_MODE=disabled` to disable logging.

To start training, download the [CelebA (Align&Cropped Images)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset and extract it to `data/celeba`. Then run:

```bash
python train.py --cuda --batch-size=128 --epochs=100 --fraction=0.25
```

This loads up 1/4 of the CelebA dataset, which approximates to 50,000 images. For 100 epochs, this trains for a little over an hour with default parameters on a single RTX 3090 (~50 seconds each epoch).

## Results

## Notes on training

The model proposed in the paper highly resembles the [DCGAN](https://dblp.org/rec/journals/corr/RadfordMC15.html) architecture, but with a few differences:

| Model | Same as DCGAN's ... | What changes?                                                                                  | Why                                                                        | Role         |
| ----- | ------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------ |
| Alice | Generator           | First layer swapped to a Linear layer.                                                         | So that the secret message can be embedded into the flattened cover image. | Encoder      |
| Bob   | Discriminator       | Final layer swapped to a Linear layer with an output of the same length as the secret message. | So that Bob can decode the secret message embedded by Alice.               | Decoder      |
| Eve   | Discriminator       | Final layer swapped to a Linear layer with an output channel of one.                           | So that Eve can distinguish cover images from stego images.                | Steganalyzer |

Changes to the training procedure that I had to make to get the model to train:

1. Learning rate is set to 1e-4 instead of 2e-4.
2. Input `image_size` is changed to 109 and `output_size` is changed to 64.
3. SGD is used for optimizing Eve instead of Adam: as Eve was getting too good (discriminator's loss drops to 0 very quickly).

The paper trained for 500 epochs, but I found that losses started to converge already after 100 epochs.

## Related

The author's original TensorFlow implementation is available at [jhayes14/advsteg](https://github.com/jhayes14/advsteg).

## Citation

```bibtex
@inproceedings{NIPS2017_fe2d0103,
    author = {Hayes, Jamie and Danezis, George},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {Generating steganographic images via adversarial training},
    url = {https://proceedings.neurips.cc/paper/2017/file/fe2d010308a6b3799a3d9c728ee74244-Paper.pdf},
    volume = {30},
    year = {2017}
}
```
