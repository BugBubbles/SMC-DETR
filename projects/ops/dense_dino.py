# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Optional
import torch
from torch import Tensor, nn
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType
from mmdet.models import DinoTransformerDecoder
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.structures import OptSampleList, SampleList
from mmdet.models.layers import coordinate_to_encoding
from .nms_dino import DINOWithNMSAlign, Branch, DINOHeadWithNMS
from mmdet.models.layers import inverse_sigmoid
from mmdet.models import SinePositionalEncoding, DeformableDetrTransformerEncoder


class DsDinoTransformerDecoder(DinoTransformerDecoder):
    def forward(
        self,
        query: Tensor,
        value: Tensor,
        key_padding_mask: Tensor,
        self_attn_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        reg_branches: nn.ModuleList,
        **kwargs,
    ) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        for l in range(self.num_layers):
            for lid, layer in enumerate(self.layers[l:]):
                if reference_points.shape[-1] == 4:
                    reference_points_input = (
                        reference_points[:, :, None]
                        * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                    )
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = (
                        reference_points[:, :, None] * valid_ratios[:, None]
                    )

                query_sine_embed = coordinate_to_encoding(
                    reference_points_input[:, :, 0, :]
                )
                query_pos = self.ref_point_head(query_sine_embed)

                query = layer(
                    query,
                    query_pos=query_pos,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points_input,
                    **kwargs,
                )

                if reg_branches is not None:
                    tmp = reg_branches[lid](query)
                    assert reference_points.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points, eps=1e-3
                    )
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()

                if self.return_intermediate:
                    intermediate.append(self.norm(query))
                    intermediate_reference_points.append(new_reference_points)
                    # NOTE this is for the "Look Forward Twice" module,
                    # in the DeformDETR, reference_points was appended.
            if not self.training and l == 0:
                break
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return query, reference_points


@MODELS.register_module()
class DsDINOWithNMSAlign(DINOWithNMSAlign):
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DsDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            f"embed_dims should be exactly 2 times of num_feats. "
            f"Found {self.embed_dims} and {num_feats}."
        )

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)


@MODELS.register_module()
class DsDETRHeadWithNMS(DINOHeadWithNMS):
    def forward(
        self, hidden_states: Tensor, references: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        layer_id = 0
        for l in range(len(self.cls_branches)):
            for cls_branch, reg_branch in zip(
                self.cls_branches[l:-1], self.reg_branches[l:-1]
            ):
                reference = inverse_sigmoid(references[layer_id])
                # NOTE The last reference will not be used.
                hidden_state = hidden_states[layer_id]
                outputs_class = cls_branch(hidden_state)
                tmp_reg_preds = reg_branch(hidden_state)
                if reference.shape[-1] == 4:
                    # When `layer` is 0 and `as_two_stage` of the detector
                    # is `True`, or when `layer` is greater than 0 and
                    # `with_box_refine` of the detector is `True`.
                    tmp_reg_preds += reference
                else:
                    # When `layer` is 0 and `as_two_stage` of the detector
                    # is `False`, or when `layer` is greater than 0 and
                    # `with_box_refine` of the detector is `False`.
                    assert reference.shape[-1] == 2
                    tmp_reg_preds[..., :2] += reference
                outputs_coord = tmp_reg_preds.sigmoid()
                all_layers_outputs_classes.append(outputs_class)
                all_layers_outputs_coords.append(outputs_coord)
                layer_id += 1
            if not self.training and l == 0:
                break
        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords


from .kv_deform_attn import KVDeformableDetrTransformerEncoder


@MODELS.register_module()
class DsDINOWithNMSAlignWithKV(DsDINOWithNMSAlign):
    def __init__(self, *args, **kwargs):
        self.encoder_cfg = kwargs["encoder"]
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        super()._init_layers()
        self.encoder = KVDeformableDetrTransformerEncoder(**self.encoder_cfg)
