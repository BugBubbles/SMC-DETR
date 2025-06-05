from .smc_detr import SMC_DETRHead, SMC_DETR
from .ds_smc_detr import DS_SMC_DETRHead, DS_SMC_DETR
import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '.'))
from .utils import *