# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor, nn
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import (
    bbox_cxcywh_to_xyxy,
    bbox_overlaps,
    bbox_xyxy_to_cxcywh,
)
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean, ConfigType
from mmdet.models.losses import QualityFocalLoss
from mmdet.models.utils import multi_apply
from mmdet.models import DeformableDETRHead
from sklearn.cluster import KMeans, DBSCAN


@MODELS.register_module()
class DEMOHead(DeformableDETRHead):
    r"""Head of the DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2203.03605>`_ .
    """

    def __init__(
        self,
        *args,
        loss_iou_reg: ConfigType = dict(
            type="CrossEntropyLoss",
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0,
        ),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_iou_reg = MODELS.build(loss_iou_reg)

    def loss(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        enc_outputs_class: Tensor,
        enc_outputs_coord: Tensor,
        batch_data_samples: SampleList,
        dn_meta: Dict[str, int],
    ) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries_total, 4) and each `inter_reference` has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_outputs_coord (Tensor): The proposal generate from the
                encode feature map, has shape (bs, num_feat_points, 4) with the
                last dimension arranged as (cx, cy, w, h).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (
            enc_outputs_class,
            enc_outputs_coord,
            batch_gt_instances,
            batch_img_metas,
            dn_meta,
        )
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # extract denoising and matching part of outputs
        (
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
        ) = self.split_outputs(
            all_layers_cls_scores,
            all_layers_bbox_preds,
            dn_meta,
        )

        loss_dict = dict()
        assert batch_gt_instances_ignore is None, (
            f"{self.__class__.__name__} only supports "
            "for batch_gt_instances_ignore setting to None."
        )

        losses_cls, losses_bbox, losses_iou, loss_iou_reg = multi_apply(
            self.loss_by_feat_single,
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
        )

        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]
        loss_dict["loss_iou"] = losses_iou[-1]
        loss_dict["loss_iou_reg"] = loss_iou_reg[-1]
        # loss from other decoder layers
        for num_dec_layer, (
            loss_cls_i,
            loss_bbox_i,
            loss_iou_i,
            loss_iou_reg_i,
        ) in enumerate(
            zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1], loss_iou_reg[:-1])
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            loss_dict[f"d{num_dec_layer}.loss_iou"] = loss_iou_i
            loss_dict[f"d{num_dec_layer}.loss_iou_reg"] = loss_iou_reg_i
        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou, enc_losses_iou_reg = (
                self.loss_by_feat_single(
                    enc_cls_scores,
                    enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas,
                )
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox
            loss_dict["enc_loss_iou"] = enc_losses_iou
            loss_dict["enc_loss_iou_reg"] = enc_losses_iou_reg

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_bbox, dn_losses_iou, dn_losses_iou_reg = (
                self.loss_dn(
                    all_layers_denoising_cls_scores,
                    all_layers_denoising_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas,
                    dn_meta=dn_meta,
                )
            )
            # collate denoising loss
            loss_dict["dn_loss_cls"] = dn_losses_cls[-1]
            loss_dict["dn_loss_bbox"] = dn_losses_bbox[-1]
            loss_dict["dn_loss_iou"] = dn_losses_iou[-1]
            loss_dict["dn_loss_iou_reg"] = dn_losses_iou_reg[-1]
            for num_dec_layer, (
                loss_cls_i,
                loss_bbox_i,
                loss_iou_i,
                loss_iou_reg_i,
            ) in enumerate(
                zip(
                    dn_losses_cls[:-1],
                    dn_losses_bbox[:-1],
                    dn_losses_iou[:-1],
                    dn_losses_iou_reg[:-1],
                )
            ):
                loss_dict[f"d{num_dec_layer}.dn_loss_cls"] = loss_cls_i
                loss_dict[f"d{num_dec_layer}.dn_loss_bbox"] = loss_bbox_i
                loss_dict[f"d{num_dec_layer}.dn_loss_iou"] = loss_iou_i
                loss_dict[f"d{num_dec_layer}.dn_loss_iou_reg"] = loss_iou_reg_i
        return loss_dict

    def loss_by_feat_single(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list, bbox_preds_list, batch_gt_instances, batch_img_metas
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0).flatten()
        label_weights = torch.cat(label_weights_list, 0).flatten()
        bbox_targets = torch.cat(bbox_targets_list, 0).flatten(0, 1)
        bbox_weights = torch.cat(bbox_weights_list, 0).flatten(0, 1)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
            scores = label_weights.new_zeros(labels.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            scores[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True
            )
            loss_cls = self.loss_cls(
                cls_scores, (labels, scores), label_weights, avg_factor=cls_avg_factor
            )
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor
            )

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w = img_meta["img_shape"]
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).repeat(
                *bbox_pred.shape[:2], 1
            )
            factors.append(factor)
        factors = torch.cat(factors, 0).flatten(0, 1)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos
        )

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos
        )

        # regression iou loss
        loss_iou_reg = self.loss_iou_reg(
            bbox_preds,
            bbox_targets,
            bbox_weights[:, 0],
            avg_factor=num_total_pos,
        )
        return loss_cls, loss_bbox, loss_iou, loss_iou_reg

    def loss_dn(
        self,
        all_layers_denoising_cls_scores: Tensor,
        all_layers_denoising_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
    ) -> Tuple[List[Tensor]]:
        """Calculate denoising loss.

        Args:
            all_layers_denoising_cls_scores (Tensor): Classification scores of
                all decoder layers in denoising part, has shape (
                num_decoder_layers, bs, num_denoising_queries,
                cls_out_channels).
            all_layers_denoising_bbox_preds (Tensor): Regression outputs of all
                decoder layers in denoising part. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and has shape
                (num_decoder_layers, bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[List[Tensor]]: The loss_dn_cls, loss_dn_bbox, and loss_dn_iou
            of each decoder layers.
        """
        return multi_apply(
            self._loss_dn_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta,
        )

    def _loss_dn_single(
        self,
        dn_cls_scores: Tensor,
        dn_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
    ) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        cls_reg_targets = self.get_dn_targets(
            batch_gt_instances, batch_img_metas, dn_meta
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        labels = (
            torch.stack(labels_list)
            .unsqueeze(1)
            .repeat(1, dn_meta["num_layers"], 1)
            .flatten()
        )
        label_weights = (
            torch.stack(label_weights_list)
            .unsqueeze(1)
            .repeat(1, dn_meta["num_layers"], 1)
            .flatten()
        )
        bbox_targets = (
            torch.stack(bbox_targets_list)
            .unsqueeze(1)
            .repeat(1, dn_meta["num_layers"], 1, 1)
            .flatten(0, 2)
        )
        bbox_weights = (
            torch.stack(bbox_weights_list)
            .unsqueeze(1)
            .repeat(1, dn_meta["num_layers"], 1, 1)
            .flatten(0, 2)
        )

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, QualityFocalLoss):
                bg_class_ind = self.num_classes
                pos_inds = (
                    ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
                )
                scores = label_weights.new_zeros(labels.shape)
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
                pos_bbox_pred = dn_bbox_preds.reshape(-1, 4)[pos_inds]
                pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
                scores[pos_inds] = bbox_overlaps(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_targets,
                    is_aligned=True,
                )
                loss_cls = self.loss_cls(
                    cls_scores,
                    (labels, scores),
                    weight=label_weights,
                    avg_factor=cls_avg_factor,
                )
            else:
                loss_cls = self.loss_cls(
                    cls_scores, labels, label_weights, avg_factor=cls_avg_factor
                )
        else:
            loss_cls = torch.zeros(1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta["img_shape"]
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).repeat(
                *bbox_pred.shape[:2], 1
            )
            factors.append(factor)
        factors = torch.cat(factors, 0).flatten(0, 1)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos
        )

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos
        )

        # regression iou loss
        loss_iou_reg = self.loss_iou_reg(
            bbox_preds,
            bbox_targets,
            bbox_weights[:, 0],
            avg_factor=num_total_pos,
        )
        return loss_cls, loss_bbox, loss_iou, loss_iou_reg

    def get_dn_targets(
        self,
        batch_gt_instances: InstanceList,
        batch_img_metas: dict,
        dn_meta: Dict[str, int],
    ) -> tuple:
        """Get targets in denoising part for a batch of images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_dn_targets_single,
            batch_gt_instances,
            batch_img_metas,
            dn_meta=dn_meta,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def _get_dn_targets_single(
        self, gt_instances: InstanceData, img_meta: dict, dn_meta: Dict[str, int]
    ) -> tuple:
        """Get targets in denoising part for one image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_groups = dn_meta["num_denoising_groups"]
        num_denoising_queries = dn_meta["num_denoising_queries"]
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2

        # label targets
        labels = gt_bboxes.new_full(
            (num_denoising_queries,), self.num_classes, dtype=torch.long
        )
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w = img_meta["img_shape"]

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized).repeat(
            [num_groups, 1]
        )
        bbox_targets[pos_inds] = gt_bboxes_targets

        # # filter badly labelled bboxes and labels
        # shape_ratio = gt_bboxes_targets[..., 2].div(gt_bboxes_targets[..., 3].add(1e-9))
        # dbscan = DBSCAN(eps=0.1, min_samples=1).fit(shape_ratio.view(-1, 1).cpu())
        # # reweight by the number of points in the cluster
        # # The most common label set to 1, the others by their proportion
        # # to the most common label
        # labels_ = torch.tensor(dbscan.labels_).to(labels.device)
        # unique_, counts = labels_.unique(return_counts=True)
        # weights = torch.ones_like(labels_)
        # for i in unique_:
        #     weights[labels_ == i] = counts[i]
        # weights = weights.div(weights.max())

        # bbox_weights[pos_inds] = weights.unsqueeze(-1)
        # label_weights[pos_inds] = weights

        return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

    def _get_targets_single(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
    ) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta["img_shape"]
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(1)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = map(
            lambda x, y: InstanceData(scores=x, bboxes=y), cls_score, bbox_pred
        )
        # assigner and sampler
        assign_results = map(
            lambda x: self.assigner.assign(
                pred_instances=x,
                gt_instances=gt_instances,
                img_meta=img_meta,
            ),
            pred_instances,
        )
        return map(
            torch.stack,
            multi_apply(
                self._get_batch_targets_single,
                assign_results,
                bbox_pred,
                gt_instances=gt_instances,
                factor=factor,
                num_bboxes=num_bboxes,
            ),
        )

    def _get_batch_targets_single(
        self, assign_result, bbox_pred, *, gt_instances, factor, num_bboxes
    ):
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        # # filter badly labelled bboxes and labels
        # shape_ratio = pos_gt_bboxes_targets[..., 2].div(
        #     pos_gt_bboxes_targets[..., 3].add(1e-9)
        # )
        # dbscan = DBSCAN(eps=0.1, min_samples=1).fit(shape_ratio.view(-1, 1).cpu())
        # ## reweight by the number of points in the cluster
        # ## The most common label set to 1, the others by their proportion
        # ## to the most common label
        # labels_ = torch.tensor(dbscan.labels_).to(labels.device)
        # unique_, counts = labels_.unique(return_counts=True)
        # weights = torch.ones_like(labels_)
        # for i in unique_:
        #     weights[labels_ == i] = counts[i]
        # weights = weights.div(weights.max())

        # bbox_weights[pos_inds] = weights.unsqueeze(-1)
        # label_weights[pos_inds] = weights
        return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

    @staticmethod
    def split_outputs(
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        dn_meta: Dict[str, int],
    ) -> Tuple[Tensor]:
        """Split outputs of the denoising part and the matching part.

        For the total outputs of `num_queries_total` length, the former
        `num_denoising_queries` outputs are from denoising queries, and
        the rest `num_matching_queries` ones are from matching queries,
        where `num_queries_total` is the sum of `num_denoising_queries` and
        `num_matching_queries`.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'.

        Returns:
            Tuple[Tensor]: a tuple containing the following outputs.

            - all_layers_matching_cls_scores (Tensor): Classification scores
              of all decoder layers in matching part, has shape
              (num_decoder_layers, bs, num_matching_queries, cls_out_channels).
            - all_layers_matching_bbox_preds (Tensor): Regression outputs of
              all decoder layers in matching part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_matching_queries, 4).
            - all_layers_denoising_cls_scores (Tensor): Classification scores
              of all decoder layers in denoising part, has shape
              (num_decoder_layers, bs, num_denoising_queries,
              cls_out_channels).
            - all_layers_denoising_bbox_preds (Tensor): Regression outputs of
              all decoder layers in denoising part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_denoising_queries, 4).
        """
        if dn_meta is not None:
            num_denoising_queries = dn_meta["num_denoising_queries"]
            all_layers_denoising_cls_scores = all_layers_cls_scores[
                ..., :num_denoising_queries, :
            ]
            all_layers_denoising_bbox_preds = all_layers_bbox_preds[
                ..., :num_denoising_queries, :
            ]
            all_layers_matching_cls_scores = all_layers_cls_scores[
                ..., num_denoising_queries:, :
            ]
            all_layers_matching_bbox_preds = all_layers_bbox_preds[
                ..., num_denoising_queries:, :
            ]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
        )

    def predict_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        batch_img_metas: List[Dict],
        rescale: bool = False,
    ) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (num_decoder_layers, bs, num_queries,
                4) with the last dimension arranged as (cx, cy, w, h).
            batch_img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Default `False`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        cls_scores = all_layers_cls_scores[-1, :, -1]
        bbox_preds = all_layers_bbox_preds[-1, :, -1]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                cls_score, bbox_pred, img_meta, rescale
            )
            result_list.append(results)
        return result_list


@MODELS.register_module()
class DEMOHead_modified(DEMOHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self.train_cfg, "weights"):
            weights = self.train_cfg["weights"]
        else:
            weights = [1.0]
        self.register_buffer("weights", torch.tensor(weights))

    def loss_by_feat_single(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list, bbox_preds_list, batch_gt_instances, batch_img_metas
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0).flatten()
        label_weights = (
            torch.stack(label_weights_list).mul(self.weights[None, :, None]).flatten()
        )
        bbox_targets = torch.cat(bbox_targets_list, 0).flatten(0, 1)
        bbox_weights = (
            torch.stack(bbox_weights_list)
            .mul(self.weights[None, :, None, None])
            .flatten(0, 2)
        )

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = (
            label_weights[bbox_weights.gt(0).all(1)].sum()
            + num_total_neg * self.bg_cls_weight
        )
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
            scores = label_weights.new_zeros(labels.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            scores[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True
            )
            loss_cls = self.loss_cls(
                cls_scores, (labels, scores), label_weights, avg_factor=cls_avg_factor
            )
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor
            )

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w = img_meta["img_shape"]
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).repeat(
                *bbox_pred.shape[:2], 1
            )
            factors.append(factor)
        factors = torch.cat(factors, 0).flatten(0, 1)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos
        )

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos
        )

        # regression iou loss
        loss_iou_reg = self.loss_iou_reg(
            bbox_preds,
            bbox_targets,
            bbox_weights[:, 0],
            avg_factor=num_total_pos,
        )
        return loss_cls, loss_bbox, loss_iou, loss_iou_reg

    def _loss_dn_single(
        self,
        dn_cls_scores: Tensor,
        dn_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
    ) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        cls_reg_targets = self.get_dn_targets(
            batch_gt_instances, batch_img_metas, dn_meta
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        labels = (
            torch.stack(labels_list)
            .unsqueeze(1)
            .repeat(1, dn_meta["num_layers"], 1)
            .flatten()
        )
        label_weights = (
            torch.stack(label_weights_list)
            .unsqueeze(1)
            .mul(self.weights[None, :, None])
            .flatten()
        )
        bbox_targets = (
            torch.stack(bbox_targets_list)
            .unsqueeze(1)
            .repeat(1, dn_meta["num_layers"], 1, 1)
            .flatten(0, 2)
        )
        bbox_weights = (
            torch.stack(bbox_weights_list)
            .unsqueeze(1)
            .mul(self.weights[None, :, None, None])
            .flatten(0, 2)
        )

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = (
            label_weights[bbox_weights.gt(0).all(1)].sum()
            + num_total_neg * self.bg_cls_weight
        )
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, QualityFocalLoss):
                bg_class_ind = self.num_classes
                pos_inds = (
                    ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
                )
                scores = label_weights.new_zeros(labels.shape)
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
                pos_bbox_pred = dn_bbox_preds.reshape(-1, 4)[pos_inds]
                pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
                scores[pos_inds] = bbox_overlaps(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_targets,
                    is_aligned=True,
                )
                loss_cls = self.loss_cls(
                    cls_scores,
                    (labels, scores),
                    weight=label_weights,
                    avg_factor=cls_avg_factor,
                )
            else:
                loss_cls = self.loss_cls(
                    cls_scores, labels, label_weights, avg_factor=cls_avg_factor
                )
        else:
            loss_cls = torch.zeros(1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta["img_shape"]
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).repeat(
                *bbox_pred.shape[:2], 1
            )
            factors.append(factor)
        factors = torch.cat(factors, 0).flatten(0, 1)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos
        )

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos
        )

        # regression iou loss
        loss_iou_reg = self.loss_iou_reg(
            bbox_preds,
            bbox_targets,
            bbox_weights[:, 0],
            avg_factor=num_total_pos,
        )
        return loss_cls, loss_bbox, loss_iou, loss_iou_reg
