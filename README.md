#  FEATURE COMPARE KD-SEG

Thanks to CIRKD ([Cross-Image Relational Knowledge Distillation for Semantic Segmentation](https://arxiv.org/pdf/2204.06986.pdf))
We propose a new kd seg structure base on their code



## Requirement


Ubuntu 18.04 LTS

Python 3.8 ([Anaconda](https://www.anaconda.com/) is recommended)

CUDA 11.1

PyTorch 1.8.0

NCCL for CUDA 11.1

Install python packages:
```
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
```

Backbones pretrained on ImageNet:

| CNN | Transformer |
| -- | -- |
|[resnet101-imagenet.pth](https://drive.google.com/file/d/1V8-E4wm2VMsfnNiczSIDoSM7JJBMARkP/view?usp=sharing)| [mit_b0.pth](https://pan.baidu.com/s/1Figp042rc9VNtPc_fkNW3g?pwd=swor )|
|[resnet18-imagenet.pth](https://drive.google.com/file/d/1_i0n3ZePtQuh66uQIftiSwN7QAUlFb8_/view?usp=sharing) | [mit_b1.pth](https://pan.baidu.com/s/1OUblLHQbq18DvXGzRU58jA?pwd=03yb)|
|[mobilenetv2-imagenet.pth](https://drive.google.com/file/d/12EDZjDSCuIpxPv-dkk1vrxA7ka0b0Yjv/view?usp=sharing) | [mit_b4.pth](https://pan.baidu.com/s/1j8pXjZZ-YSi2JXpsaQSSTQ?pwd=cvpd )|


Support datasets:

| Dataset | Train Size | Val Size | Test Size | Class |
| -- | -- | -- |-- |-- |
| Cityscapes | 2975 | 500 | 1525 |19|
| Pascal VOC Aug | 10582 | 1449 | -- | 21 |
| CamVid | 367 | 101 | 233 | 11 |
| ADE20K | 20210 | 2000 | -- | 150 |
| COCO-Stuff-164K | 118287 | 5000 |-- | 182 |






