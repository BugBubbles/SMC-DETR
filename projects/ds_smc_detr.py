# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType
from mmcv.ops import MultiScaleDeformableAttention, batched_nms
from mmdet.models import DDQTransformerDecoder
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.structures import OptSampleList, SampleList
from mmdet.models.utils import align_tensor
from .smc_detr import SMC_DETR, Branch
from mmdet.models.layers import inverse_sigmoid
from mmdet.models import DeformableDETR, DDQDETRHead, DeformableDETRHead
import torchvision
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean, ConfigType
from mmdet.models.losses import QualityFocalLoss
from .smc_detr import SEDADetrTransformerEncoder


@MODELS.register_module()
class DS_SMC_DETR(SMC_DETR):
    def __init__(
        self,
        *args,
        dense_topk_ratio: float = 1.5,
        dqs_cfg: OptConfigType = dict(type="nms", iou_threshold=0.8),
        **kwargs,
    ):
        self.dense_topk_ratio = dense_topk_ratio
        self.decoder_cfg = kwargs["decoder"]
        self.dqs_cfg = dqs_cfg
        self.encoder_cfg = kwargs["encoder"]
        super().__init__(*args, **kwargs)

        # a share dict in all moduls
        # pass some intermediate results and config parameters
        cache_dict = dict()
        for m in self.modules():
            m.cache_dict = cache_dict
        # first element is the start index of matching queries
        # second element is the number of matching queries
        self.cache_dict["dis_query_info"] = [0, 0]

        # mask for distinct queries in each decoder layer
        self.cache_dict["distinct_query_mask"] = []
        # pass to decoder do the dqs
        self.cache_dict["cls_branches"] = self.bbox_head.cls_branches
        # Used to construct the attention mask after dqs
        self.cache_dict["num_heads"] = self.encoder.layers[0].self_attn.num_heads
        # pass to decoder to do the dqs
        self.cache_dict["dqs_cfg"] = self.dqs_cfg

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        super(SMC_DETR, self)._init_layers()
        self.decoder = DDQTransformerDecoder(**self.decoder_cfg)
        self.encoder = SEDADetrTransformerEncoder(**self.encoder_cfg)
        self.query_embedding = None
        self.query_map = nn.Linear(self.embed_dims, self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.normal_(self.level_embed)

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `memory`, and `reference_points`.

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
              `dense_topk_score`, `dense_topk_coords`,
              and `dn_meta`, when `self.training` is `True`, else is empty.
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
        proposals = enc_outputs_coord_unact.sigmoid()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        scores = enc_outputs_class.max(-1)[0].sigmoid()
        topk_indices = torch.topk(scores, k=self.num_queries, dim=1)[1].squeeze(-1)
        topk_score = torch.gather(
            enc_outputs_class,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features),
        )
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4)
        )

        map_memory = torch.gather(memory, 1, topk_indices.unsqueeze(-1).repeat(1, 1, c))
        query = self.query_map(map_memory.detach())
        topk = self.num_queries
        if self.training:
            # aux dense branch particularly in DDQ DETR, which doesn't exist
            #   in DINO.
            # -1 is the aux head for the encoder
            dense_enc_outputs_class = self.bbox_head.cls_branches[-1](output_memory)
            dense_enc_outputs_class = (
                dense_enc_outputs_class.sigmoid() * enc_outputs_pss.sigmoid()
            )
            dense_enc_outputs_class = inverse_sigmoid(dense_enc_outputs_class)
            dense_enc_outputs_coord_unact = (
                self.bbox_head.reg_branches[-1](output_memory) + output_proposals
            )
            # aux dense branch particularly in DDQ DETR, which doesn't exist
            #   in DINO.
            dense_proposals = dense_enc_outputs_coord_unact.sigmoid()
            dense_proposals = bbox_cxcywh_to_xyxy(dense_proposals)
            if torchvision._is_tracing():
                topk = torch.min(torch.tensor(self.num_queries * 4), n)
            else:
                topk = min(self.num_queries * 4, n)
            topk_score_nms, topk_indices_nms = torch.topk(
                enc_outputs_class.max(-1)[0], k=topk, dim=1
            )
            # add nms procedure
            topk_indices_nms = self.nms_on_topk_index(
                topk_score_nms,
                topk_indices_nms,
                spatial_shapes,
                level_start_index,
                ratio=2,
            )

            if torchvision._is_tracing():
                topk = torch.min(
                    torch.tensor(self.num_queries * 4 * self.dense_topk_ratio).long(), n
                )
            else:
                topk = min(int(self.num_queries * 4 * self.dense_topk_ratio), n)
            dense_scores, dense_indices = torch.topk(
                dense_enc_outputs_class.max(-1)[0].sigmoid(), k=topk, dim=1
            )
            # add nms procedure
            dense_indices = self.nms_on_topk_index(
                dense_scores, dense_indices, spatial_shapes, level_start_index, ratio=2
            )
            dense_scores = dense_enc_outputs_class.max(-1)[0].sigmoid()

            num_imgs = len(scores)
            topk_indices_nms_ = []

            dense_topk_score = []
            dense_topk_coords_unact = []
            dense_query = []
            dense_topk_indices = []

            topk = self.num_queries
            dense_topk = int(topk * self.dense_topk_ratio)
            for img_id in range(num_imgs):
                single_proposals = proposals[img_id][topk_indices_nms[img_id]]
                single_scores = scores[img_id][topk_indices_nms[img_id]]

                # `batched_nms` of class scores and bbox coordinations is used
                #   particularly by DDQ DETR for region proposal generation,
                #   instead of `topk` of class scores by DINO.
                _, keep_idxs = batched_nms(
                    single_proposals,
                    single_scores,
                    torch.ones(len(single_scores), device=single_scores.device),
                    self.cache_dict["dqs_cfg"],
                )

                keep_idxs = topk_indices_nms[img_id][keep_idxs]
                topk_indices_nms_.append(keep_idxs[:topk])

                # aux dense branch particularly in DDQ DETR, which doesn't
                #   exist in DINO.
                dense_single_proposals = dense_proposals[img_id][dense_indices[img_id]]
                dense_single_scores = dense_scores[img_id][dense_indices[img_id]]
                # sort according the score
                # Only sort by classification score, neither nms nor topk is
                #   required. So input parameter `nms_cfg` = None.
                _, dense_keep_idxs = batched_nms(
                    dense_single_proposals,
                    dense_single_scores,
                    dense_single_scores.new_ones(len(dense_single_scores)).long(),
                    None,
                )

                dense_keep_idxs = dense_indices[img_id][dense_keep_idxs]
                dense_topk_score.append(
                    dense_enc_outputs_class[img_id][dense_keep_idxs][:dense_topk]
                )
                dense_topk_coords_unact.append(
                    dense_enc_outputs_coord_unact[img_id][dense_keep_idxs][:dense_topk]
                )

                # Instead of initializing the content part with transformed
                #   coordinates in Deformable DETR, we fuse the feature map
                #   embedding of distinct positions as the content part, which
                #   makes the initial queries more distinct.

                map_memory = self.query_map(memory[img_id].detach())
                # aux dense branch particularly in DDQ DETR, which doesn't
                # exist in DINO.
                dense_query.append(map_memory[dense_keep_idxs][:dense_topk])
                dense_topk_indices.append(dense_keep_idxs[:dense_topk])
            topk_indices_nms = align_tensor(topk_indices_nms_, topk)

            dense_topk_score = align_tensor(dense_topk_score)
            dense_topk_coords_unact = align_tensor(dense_topk_coords_unact)
            dense_topk_indices = align_tensor(dense_topk_indices)
            dense_query = align_tensor(dense_query)
            num_dense_queries = dense_query.size(1)

            query = torch.cat([query, dense_query], dim=1)
            topk_coords_unact = torch.cat(
                [topk_coords_unact, dense_topk_coords_unact], dim=1
            )

        topk_coords = topk_coords_unact.sigmoid().clone()
        if self.training:
            dense_topk_coords = topk_coords[:, -num_dense_queries:]
            topk_coords = topk_coords[:, :-num_dense_queries]

        topk_coords_unact = topk_coords_unact.detach()

        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(
                batch_data_samples
            )
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)

            # Update `dn_mask` to add mask for dense queries.
            ori_size = dn_mask.size(-1)
            new_size = dn_mask.size(-1) + num_dense_queries
            new_dn_mask = dn_mask.new_ones((new_size, new_size)).bool()
            dense_mask = torch.zeros(num_dense_queries, num_dense_queries).bool()
            self.cache_dict["dis_query_info"] = [dn_label_query.size(1), topk]

            new_dn_mask[ori_size:, ori_size:] = dense_mask
            new_dn_mask[:ori_size, :ori_size] = dn_mask
            dn_meta["num_dense_queries"] = num_dense_queries
            dn_mask = new_dn_mask
            self.cache_dict["num_dense_queries"] = num_dense_queries
            self.decoder.aux_reg_branches = self.bbox_head.aux_reg_branches

        else:
            self.cache_dict["dis_query_info"] = [0, topk]
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None

        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
        )
        head_inputs_dict = (
            dict(
                enc_outputs_class=topk_score,
                enc_outputs_coord=topk_coords,
                aux_enc_outputs_class=dense_topk_score,
                aux_enc_outputs_coord=dense_topk_coords,
                enc_outputs_pss=enc_outputs_pss,
                topk_indices=topk_indices_nms,
                dense_topk_indices=dense_topk_indices,
                dn_meta=dn_meta,
            )
            if self.training
            else dict()
        )

        return decoder_inputs_dict, head_inputs_dict


@MODELS.register_module()
class DS_SMC_DETRHead(DDQDETRHead):
    def __init__(
        self,
        *args,
        loss_rank: ConfigType = None,
        loss_dense_rank: ConfigType = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_rank = MODELS.build(loss_rank) if loss_rank else None
        self.loss_dense_rank = (
            MODELS.build(loss_dense_rank) if loss_dense_rank else None
        )

    def _init_layers(self):
        super()._init_layers()
        self.encode_pss_branch = Branch(self.embed_dims, 1)

    def loss(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        enc_outputs_class: Tensor,
        enc_outputs_coord: Tensor,
        enc_outputs_pss: Tensor,
        batch_data_samples: SampleList,
        dn_meta: Dict[str, int],
        aux_enc_outputs_class: Tensor = None,
        aux_enc_outputs_coord: Tensor = None,
        topk_indices: Tensor = None,
        dense_topk_indices: Tensor = None,
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
            enc_outputs_pss,
            topk_indices,
            dense_topk_indices,
        )
        losses = self.loss_by_feat(*loss_inputs)
        # dense distinct query loss
        aux_enc_outputs_coord_list = []
        for img_id in range(len(aux_enc_outputs_coord)):
            det_bboxes = aux_enc_outputs_coord[img_id]
            img_shape = batch_img_metas[img_id]["img_shape"]
            det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
            det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
            aux_enc_outputs_coord_list.append(det_bboxes)
        aux_enc_outputs_coord = torch.stack(aux_enc_outputs_coord_list)
        aux_loss = self.aux_loss_for_dense.loss(
            aux_enc_outputs_class.sigmoid(),
            aux_enc_outputs_coord,
            [item.bboxes for item in batch_gt_instances],
            [item.labels for item in batch_gt_instances],
            batch_img_metas,
        )
        for k, v in aux_loss.items():
            losses[f"aux_enc_{k}"] = v

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
        enc_pss_scores: Tensor = None,
        topk_indices: Tensor = None,
        dense_topk_indices: Tensor = None,
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

        num_dense_queries = dn_meta["num_dense_queries"]
        num_layer = all_layers_matching_bbox_preds.size(0)
        dense_all_layers_matching_cls_scores = all_layers_matching_cls_scores[
            :, :, -num_dense_queries:  # noqa: E501
        ]  # noqa: E501
        dense_all_layers_matching_bbox_preds = all_layers_matching_bbox_preds[
            :, :, -num_dense_queries:  # noqa: E501
        ]  # noqa: E501

        all_layers_matching_cls_scores = all_layers_matching_cls_scores[
            :, :, :-num_dense_queries  # noqa: E501
        ]  # noqa: E501
        all_layers_matching_bbox_preds = all_layers_matching_bbox_preds[
            :, :, :-num_dense_queries  # noqa: E501
        ]  # noqa: E501

        loss_dict = self.loss_for_distinct_queries(
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
            (
                enc_loss_cls,
                enc_losses_bbox,
                enc_losses_iou,
                enc_loss_rank,
                enc_loss_dense_rank,
            ) = self.loss_by_feat_single_encode(
                enc_cls_scores,
                enc_bbox_preds,
                batch_gt_instances,
                batch_img_metas,
                enc_pss_scores,
                topk_indices,
                dense_topk_indices,
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox
            loss_dict["enc_loss_iou"] = enc_losses_iou
            loss_dict["enc_loss_rank"] = enc_loss_rank
            loss_dict["enc_loss_dense_rank"] = enc_loss_dense_rank

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

        for l_id in range(num_layer):
            cls_scores = dense_all_layers_matching_cls_scores[l_id].sigmoid()
            bbox_preds = dense_all_layers_matching_bbox_preds[l_id]

            bbox_preds = bbox_cxcywh_to_xyxy(bbox_preds)
            bbox_preds_list = []
            for img_id in range(len(bbox_preds)):
                det_bboxes = bbox_preds[img_id]
                img_shape = batch_img_metas[img_id]["img_shape"]
                det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
                det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
                bbox_preds_list.append(det_bboxes)
            bbox_preds = torch.stack(bbox_preds_list)
            aux_loss = self.aux_loss_for_dense.loss(
                cls_scores,
                bbox_preds,
                [item.bboxes for item in batch_gt_instances],
                [item.labels for item in batch_gt_instances],
                batch_img_metas,
            )
            for k, v in aux_loss.items():
                loss_dict[f"{l_id}_aux_{k}"] = v

        return loss_dict

    def loss_by_feat_single_encode(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        pss_scores: Tensor = None,
        topk_indices: Tensor = None,
        enc_loss_dense_rank: Tensor = None,
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
        loss_dense_rank = []
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
                # rank loss for nms
            # if enc_loss_dense_rank is not None:
            #     labels = torch.zeros_like(pss_scores.squeeze(-1)).scatter(
            #         1, enc_loss_dense_rank, 1
            #     )
            #     loss_dense_rank.append(self.loss_dense_rank(pss_scores, labels))
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
        return loss_cls, loss_bbox, loss_iou, loss_rank, loss_dense_rank
