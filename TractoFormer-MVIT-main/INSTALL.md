# Installation

## Requirements
- Python >= 3.8
- PyTorch >= 1.7, please follow PyTorch official instructions at [pytorch.org](https://pytorch.org)
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- iopath: `pip install -U iopath` or `conda install -c iopath iopath`
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation. You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- simplejson: `pip install simplejson`
- FairScale: `pip install 'git+https://github.com/facebookresearch/fairscale'`
- psutil: `pip install psutil`
- nibabel `pip install nibabel`

## MViT

```
cd mvit
python setup.py build develop
```
