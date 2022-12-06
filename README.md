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

## Results

## Notes on training

The model proposed in the paper highly resembles the [DCGAN](https://dblp.org/rec/journals/corr/RadfordMC15.html) architecture, but with a few differences:

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
