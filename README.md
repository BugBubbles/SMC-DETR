# Description

This repository is for the detection of small- and medium-sized lunar impact craters. 

# Result

This result below attain from MDCD dataset, refer to "High-resolution feature pyramid network for automatic crater detection on Mars". You can download it from their official [site](https://doi.org/10.5281/zenodo.4750929).

| epoch | mAP   |
| ----- | ----- |
| 50    | 0.321 |

## Installation

First install the `lastest` mmedetection following: [https://mmdetection.readthedocs.io/en/latest/get_started.html](https://mmdetection.readthedocs.io/en/latest/get_started.html)

Then go to `mmdetection/projects`, clone this repostry:

```bash
git clone https://github.com/BugBubbles/SMC-DETR ./SMC-DETR
```

Then you can use this repostry as any one in `projects`. Maybe you should first deteminate your datasets links in `./datasets/*.py` files. Or you can download the Dataset from ``.
