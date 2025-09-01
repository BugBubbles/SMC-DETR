# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional, Tuple, Union
from mmengine.structures import InstanceData
import mmcv
import numpy as np

from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from mmengine.dist import master_only
from mmdet.visualization import DetLocalVisualizer
from scipy.optimize import linear_sum_assignment
from mmdet.structures.bbox import bbox_overlaps
from mmdet.visualization.palette import _get_adaptive_scales, get_palette, jitter_color


@VISUALIZERS.register_module()
class DetLocalVisualizer2(DetLocalVisualizer):
    @master_only
    def add_datasample(
        self,
        name: str,
        image: np.ndarray,
        data_sample: Optional[DetDataSample] = None,
        draw_gt: bool = True,
        draw_pred: bool = True,
        show: bool = False,
        wait_time: float = 0,
        # TODO: Supported in mmengine's Viusalizer.
        out_file: Optional[str] = None,
        pred_score_thr: float = 0.05,
        iou_threshold: float = 0.3,
        step: int = 0,
    ) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        image = image.clip(0, 255).astype(np.uint8)

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_pred and data_sample is not None:
            if "pred_instances" in data_sample:
                pred_instances = data_sample.pred_instances
                pred_instances = pred_instances[pred_instances.scores > pred_score_thr]
                # 将预测结果分为两个集合：符合GT的，GT中不包含的，这里是单个类别，所以不考虑label
                # 计算预测结果与GT的匈牙利一对一分配
                if "gt_instances" in data_sample:
                    iou = bbox_overlaps(
                        data_sample.gt_instances.bboxes, pred_instances.bboxes
                    )
                    gt_indices, pred_indices = linear_sum_assignment(-iou.cpu().numpy())
                    # 保证匹配预测框与gt框所得iou应当大于pred_score_thr
                    pred_iou = iou[gt_indices, pred_indices] > iou_threshold
                    pred_indices = pred_indices[pred_iou]
                    gt_indices = gt_indices[pred_iou]
                    # pred_in_gt = pred_instances.bboxes[pred_indices]
                    pred_out_gt = pred_instances.bboxes[
                        np.setdiff1d(
                            np.arange(len(pred_instances.bboxes)), pred_indices
                        )
                    ]
                    gt_in_pred = data_sample.gt_instances.bboxes[gt_indices]
                    gt_out_pred = data_sample.gt_instances.bboxes[
                        np.setdiff1d(
                            np.arange(len(data_sample.gt_instances)), gt_indices
                        )
                    ]
                    # 不绘制gt_in_pred，为剩下的三种情况绘制对应的检测框，使用不同的颜色
                    drawn_img = self._draw_craters(image, gt_in_pred, (255, 0, 0))
                    drawn_img = self._draw_craters(drawn_img, gt_out_pred, (0, 255, 0))
                    drawn_img = self._draw_craters(drawn_img, pred_out_gt, (0, 160, 160))

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)

    def _draw_craters(
        self,
        image: np.ndarray,
        craters,
        color: tuple[int],
    ) -> np.ndarray:
        self.set_image(image)

        if craters.sum() > 0:

            centers = (craters[:, :2] + craters[:, 2:]) / 2
            radius = (craters[:, 2:] - craters[:, :2]).sum(1) / 4
            self.draw_circles(
                centers,
                radius,
                edge_colors=color,
                alpha=self.alpha,
                line_widths=self.line_width,
            )
        return self.get_image()
