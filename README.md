# NDDR-CNN-PyTorch
PyTorch Implementation of [NDDR-CNN](https://arxiv.org/abs/1801.08297)

## Usage
### Requirements
- PyTorch 1.1
- NumPy
- YACS
- PIL
- tensorboardX
- argparse
- tqdm

### Setup
The above dependencies can be installed by the following commands via pip:
```sh
$ pip install torch torchvision numpy yacs pillow tensorboardX argparse tqdm
```

### Dataset
Follow the instruction in the [official repository](https://github.com/ethanygao/NDDR-CNN) to prepare the dataset.
Then Download the converted PyTorch models from [here](https://drive.google.com/file/d/1mXsWHYlE-u4EM0Sr4XvBYcLxCFJ-F4dd/view?usp=sharing), then create a `weights` directory and unzip the models inside.

When you are all set, you should have the following file structure:
```
datasets/nyu_v2/list
datasets/nyu_v2/nyu_train_val
weights/vgg_deeplab_lfov/tf_deeplab.pth
weights/nyu_v2/tf_finetune_seg.pth
weights/nyu_v2/tf_finetune_normal.pth
```

### Training
All the arguments to train/eval an NDDR-CNN are shown in `configs/defaults.py`. The configuration files for different experiments are also provided in the `configs` directory. For example, to train an NDDR-CNN with VGG-16-shortcut architecture, simply call:

```sh
$ CUDA_VISIBLE_DEVICES=0 python train.py --config-file configs/vgg16_nddr_shortcut_sing.yaml
```

### Evaluation
To evaluate an NDDR-CNN with VGG-16-shortcut architecture, call:

```sh
$ CUDA_VISIBLE_DEVICES=0 python eval.py --config-file configs/vgg16_nddr_shortcut_sing.yaml
```

## Downloads
- Preprocessed Dataset ([link](https://www.dropbox.com/sh/e44jyh6ayuimigp/AADHlrCVnCDyTdDT9wDOy8cUa?dl=0))
- PyTorch weight initializations converted from TensorFlow pretrained models ([link](https://drive.google.com/file/d/1mXsWHYlE-u4EM0Sr4XvBYcLxCFJ-F4dd/view?usp=sharing))
- A rough script for converting TensorFlow model to PyTorch is available [here](https://gist.github.com/bhpfelix/8001f2e2c4770655e23ad0c1900f1f15)

## Reference:
- [Official Repository](https://github.com/ethanygao/NDDR-CNN)
- [Deeplab-Large-FOV](https://github.com/CoinCheung/Deeplab-Large-FOV)
