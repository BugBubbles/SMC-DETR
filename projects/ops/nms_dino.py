# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean, ConfigType
from mmdet.models.losses import QualityFocalLoss
from mmdet.models import DINOHead, DINO, DeformableDETRHead
import torchvision
from mmdet.models.layers import inverse_sigmoid


@MODELS.register_module()
class DINOWithNMS(DINO):
    def __init__(self, *args, iou_threshold=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.iou_threshold = iou_threshold

    def forward_encoder(
        self,
        feat: Tensor,
        feat_mask: Tensor,
        feat_pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
    ) -> Dict:
        output_dict = super().forward_encoder(
            feat, feat_mask, feat_pos, spatial_shapes, level_start_index, valid_ratios
        )
        output_dict.update(level_start_index=level_start_index)
        return output_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        bs, n, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )
        enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](
            output_memory
        )
        enc_outputs_coord_unact = (
            self.bbox_head.reg_branches[self.decoder.num_layers](output_memory)
            + output_proposals
        )

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        # get topk output classes and coordinates
        if torchvision._is_tracing():
            topk = torch.min(torch.tensor(self.num_queries * 4), n)
        else:
            topk = min(self.num_queries * 4, n)
        topk_score, topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=topk, dim=1
        )
        # add nms procedure
        # 主要是对坐标的NMS，与预测得到的框无关，当前通过额外的分数项分支预测各个位置上的objectness
        # 所得的效果不如该效果好，其原因是未考虑query自身的空间位置，仅把query预测得到的概率拿来强化
        # 其实与强化之前的效果差异不大，尚不能达到网格NMS的效果。SailenceDETR采用的网格NMS也仅仅是去
        # 除了同一分辨率下非常接近的query（相差一个grid的），能够保证每个query之间至少差两个grid，也
        # 仍然不是完全去除网格的背景query
        topk_indices = self.nms_on_topk_index(
            topk_score, topk_indices, spatial_shapes, level_start_index
        )

        topk_score = torch.gather(
            enc_outputs_class,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features),
        )
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4)
        )
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight.repeat(bs, 1, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(
                batch_data_samples
            )
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = (
            dict(
                enc_outputs_class=topk_score,
                enc_outputs_coord=topk_coords,
                dn_meta=dn_meta,
            )
            if self.training
            else dict()
        )
        return decoder_inputs_dict, head_inputs_dict

    @torch.no_grad()
    def nms_on_topk_index(
        self,
        topk_scores,
        topk_index,
        spatial_shapes,
        level_start_index,
    ):
        batch_size, num_topk = topk_scores.shape
        if torchvision._is_tracing():
            num_pixels = spatial_shapes.prod(-1).unbind()
        else:
            num_pixels = spatial_shapes.prod(-1).tolist()

        # flatten topk_scores and topk_index for batched_nms
        topk_scores, topk_index = map(lambda x: x.view(-1), (topk_scores, topk_index))

        # get level coordinates for queries and construct boxes for them
        level_index = torch.arange(
            level_start_index.shape[0], device=level_start_index.device
        )
        feat_width, start_index, level_idx = map(
            lambda x: self.fast_repeat_interleave(x, num_pixels)[topk_index],
            (spatial_shapes[:, 1], level_start_index, level_index),
        )
        topk_spatial_index = topk_index - start_index
        x = topk_spatial_index % feat_width
        y = torch.div(topk_spatial_index, feat_width, rounding_mode="trunc")
        coordinates = torch.stack([x - 1.0, y - 1.0, x + 1.0, y + 1.0], -1)

        # get unique idx for queries in different images and levels
        image_idx = torch.arange(batch_size).repeat_interleave(num_topk, 0)
        image_idx = image_idx.to(level_idx.device)
        idxs = level_idx + level_start_index.shape[0] * image_idx

        # perform batched_nms
        indices = torchvision.ops.batched_nms(
            coordinates, topk_scores, idxs, self.iou_threshold
        )

        # stack valid index
        results_index = []
        if torchvision._is_tracing():
            min_num = torch.tensor(self.num_queries)
        else:
            min_num = self.num_queries
        # get indices in each image
        for i in range(batch_size):
            topk_index_per_image = topk_index[indices[image_idx[indices] == i]]
            if torchvision._is_tracing():
                min_num = torch.min(topk_index_per_image.shape[0], min_num)
            else:
                min_num = min(topk_index_per_image.shape[0], min_num)
            results_index.append(topk_index_per_image)
        return torch.stack([index[:min_num] for index in results_index])

    @staticmethod
    def fast_repeat_interleave(input, repeats):
        """torch.Tensor.repeat_interleave is slow for one-dimension input for unknown reasons.
        This is a simple faster implementation. Notice the return shares memory with the input.

        :param input: input Tensor
        :param repeats: repeat numbers of each element in the specified dim
        :param dim: the dimension to repeat, defaults to None
        """
        # the following inplementation runs a little faster under one-dimension settings
        return torch.cat([aa.expand(bb) for aa, bb in zip(input, repeats)])


@MODELS.register_module()
class DINOWithNMSAlign(DINOWithNMS):
    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        bs, n, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )
        enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](
            output_memory
        )
        enc_outputs_pss = self.bbox_head.encode_pss_branch(
            output_memory.detach(), spatial_shapes
        )

        enc_outputs_class = enc_outputs_class.sigmoid() * enc_outputs_pss.sigmoid()
        enc_outputs_class = inverse_sigmoid(enc_outputs_class)
        enc_outputs_coord_unact = (
            self.bbox_head.reg_branches[self.decoder.num_layers](output_memory)
            + output_proposals
        )

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        # get topk output classes and coordinates
        if self.training:
            if torchvision._is_tracing():
                topk = torch.min(torch.tensor(self.num_queries * 4), n)
            else:
                topk = min(self.num_queries * 4, n)
            topk_score, topk_indices = torch.topk(
                enc_outputs_class.max(-1)[0], k=topk, dim=1
            )
            # add nms procedure
            topk_indices_nms = self.nms_on_topk_index(
                topk_score, topk_indices, spatial_shapes, level_start_index
            )
        topk_indices = torch.topk(enc_outputs_class, k=self.num_queries, dim=1)[
            1
        ].squeeze(-1)
        topk_score = torch.gather(
            enc_outputs_class,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features),
        )
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4)
        )
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight.repeat(bs, 1, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(
                batch_data_samples
            )
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = (
            dict(
                enc_outputs_class=topk_score,
                enc_outputs_coord=topk_coords,
                enc_outputs_pss=enc_outputs_pss,
                topk_indices=topk_indices_nms,
                dn_meta=dn_meta,
            )
            if self.training
            else dict()
        )
        return decoder_inputs_dict, head_inputs_dict


from mmcv.ops import DeformConv2dPack


class Branch(nn.Module):
    def __init__(self, embed_dims, out_channels):
        super().__init__()
        self.conv = DeformConv2dPack(embed_dims, out_channels, kernel_size=3, padding=1)
        self.out_features = out_channels

    def forward(self, inputs: Tensor, spatial_shapes: Tensor):
        # 把输入拆成金字塔形状
        outs = []
        start = 0
        bs = len(inputs)
        for h, w in spatial_shapes:
            end = start + h * w
            feat = inputs[:, start:end].transpose(1, 2).contiguous().view(bs, -1, h, w)
            start = end
            outs.append(self.conv(feat).flatten(-2).transpose(1, 2))
        return torch.cat(outs, dim=1)


@MODELS.register_module()
class DINOHeadWithNMS(DINOHead):
    def __init__(self, *args, loss_rank: ConfigType = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_rank = MODELS.build(loss_rank) if loss_rank else None

    def _init_layers(self):
        super()._init_layers()
        # self.encode_pss_branch = nn.Linear(self.embed_dims, 1)
        self.encode_pss_branch = Branch(self.embed_dims, 1)

    def init_weights(self):
        super().init_weights()
        # nn.init.constant_(self.encode_pss_branch.bias, 0.01)

    def loss(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        enc_outputs_class: Tensor,
        enc_outputs_coord: Tensor,
        enc_outputs_pss: Tensor,
        topk_indices: Tensor,
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
            enc_outputs_pss,
            topk_indices,
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
        enc_pss_scores: Tensor,
        topk_indices: Tensor,
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
        ) = self.split_outputs(all_layers_cls_scores, all_layers_bbox_preds, dn_meta)

        loss_dict = super(DeformableDETRHead, self).loss_by_feat(
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
        )
        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou, enc_loss_rank = (
                self.loss_by_feat_single_encode(
                    enc_cls_scores,
                    enc_bbox_preds,
                    enc_pss_scores,
                    topk_indices,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas,
                )
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox
            loss_dict["enc_loss_iou"] = enc_losses_iou
            loss_dict["enc_loss_rank"] = enc_loss_rank

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta,
            )
            # collate denoising loss
            loss_dict["dn_loss_cls"] = dn_losses_cls[-1]
            loss_dict["dn_loss_bbox"] = dn_losses_bbox[-1]
            loss_dict["dn_loss_iou"] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in enumerate(
                zip(dn_losses_cls[:-1], dn_losses_bbox[:-1], dn_losses_iou[:-1])
            ):
                loss_dict[f"d{num_dec_layer}.dn_loss_cls"] = loss_cls_i
                loss_dict[f"d{num_dec_layer}.dn_loss_bbox"] = loss_bbox_i
                loss_dict[f"d{num_dec_layer}.dn_loss_iou"] = loss_iou_i
        return loss_dict

    def loss_by_feat_single_encode(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        pss_scores: Tensor,
        topk_indices: Tensor,
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
        B, N, C = cls_scores.size()
        cls_scores_list = [cls_scores[i] for i in range(B)]
        bbox_preds_list = [bbox_preds[i] for i in range(B)]
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
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            pos_inds = ((labels >= 0) & (labels < C)).nonzero().squeeze(1)
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

        loss_rank = []
        # rank loss for topk
        if pss_scores is not None:
            if topk_indices is not None:
                loss_rank.append(
                    self.loss_rank(cls_scores.view(B, N, C), labels.view(B, N))
                )

                labels = torch.zeros_like(pss_scores.squeeze(-1)).scatter(
                    1, topk_indices, 1
                )
                # rank loss for altogether
                loss_rank.append(self.loss_rank(pss_scores, labels))
        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w = img_meta["img_shape"]
            factor = (
                bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
                .unsqueeze(0)
                .repeat(bbox_pred.size(0), 1)
            )
            factors.append(factor)
        factors = torch.cat(factors, 0)

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

        # nms-aligned objectness loss
        return loss_cls, loss_bbox, loss_iou, loss_rank


@MODELS.register_module()
class RankLoss(nn.Module):
    def __init__(self, delta=0.5, loss_weight=0.5, topk=100, soft=6):
        super().__init__()
        self.delta = delta
        self.loss_weight = loss_weight
        self.topk = topk
        self.soft = soft

    def forward(self, pred: Tensor, target: Tensor):
        num_classes = pred.size(-1)
        tp_scores = pred[target.ne(num_classes)]
        # mask tp scores to -inf
        a = pred.masked_fill(target.ne(num_classes).unsqueeze(-1), float("-inf"))
        fp_scores = a.topk(self.topk, dim=1).values.flatten(0, 1)
        weights = torch.ones_like(fp_scores)
        weights[fp_scores.isinf()] = 0
        loss_rank = fp_scores.unsqueeze(0) - tp_scores.unsqueeze(1) + self.delta
        loss_rank = loss_rank * weights[None]
        loss_rank = 1 / self.soft * torch.log(1 + torch.exp(self.soft * loss_rank))
        return self.loss_weight * loss_rank.mean()
