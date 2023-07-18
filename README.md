# DualConv: Dual Convolutional Kernels for Lightweight Deep Neural Networks
This is the official PyTorch implementation for [DualConv paper](https://ieeexplore.ieee.org/document/9723436). If you find that this project helps your research, please consider citing the following paper:
```
@ARTICLE{9723436,
  author={Zhong, Jiachen and Chen, Junying and Mian, Ajmal},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={DualConv: Dual Convolutional Kernels for Lightweight Deep Neural Networks}, 
  year={2022},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/TNNLS.2022.3151138}}
```
## Requirements
- Pytorch 1.1+
- Python 3.6+
## Usage
The official implementation of our proposed DualConv and the reproduced HetConv and GroupConv are both in the /kernels folder. We provide example code for applying the lightweight convolutional kernel to the VGG-16 network on the CIFAR-10 dataset.
```
# Start training with: 
cd example_usage && python CIFAR-10.py --kernel dualconv

# You can manually resume the training with: 
python CIFAR-10.py --resume --lr=0.01 --kernel dualconv
```