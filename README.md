# Description

This repository is for the detection of small- and medium-sized lunar impact craters. 

## Installation

First install the `lastest` mmedetection following: [https://mmdetection.readthedocs.io/en/latest/get_started.html](https://mmdetection.readthedocs.io/en/latest/get_started.html)

Then go to `mmdetection/projects`, clone this repostry:

```bash
git clone https://github.com/BugBubbles/SMC-DETR ./SMC-DETR
```

Then you can use this repostry as any one in `projects`. Maybe you should first deteminate your datasets links in `./datasets/*.py` files. Or you can download the Dataset from `https://www.modelscope.cn/datasets/BugBubbles/CraterDetectMoon`. Then you should change the `data_root` in `./configs/_base_/datasets/*.py` files to your local path.

## Training
You can train the model by running the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/smc-detr/no-ds/smc-detr_4sr50_8xb2-200e_bo.py
```

## Testing
You can test the model by running the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/smc-detr/no-ds/smc-detr_4sr50_8xb2-200e_bo.py work_dirs/smc-detr_4sr50_8xb2-200e_bo/latest.pth
```