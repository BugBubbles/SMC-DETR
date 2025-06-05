# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.structures import OptSampleList, SampleList
from typing import Dict, Tuple, Optional, Union

import torch
from mmcv.cnn.bricks.transformer import (
    MultiScaleDeformableAttention,
    FFN,
    MultiheadAttention,
)
from mmdet.registry import MODELS
from torch import nn, Tensor
from torch.nn import functional as F
from mmengine.model import ModuleList
from mmcv.cnn import build_norm_layer
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import torchvision
from mmdet.models.layers import inverse_sigmoid
from mmdet.models import (
    DeformableDetrTransformerEncoder,
    DinoTransformerDecoder,
    SinePositionalEncoding,
    DINO,
    DeformableDetrTransformerDecoderLayer,
    DeformableDetrTransformerEncoderLayer,
)
from ..ops.kv_deform_attn import (
    KVDeformableDetrTransformerEncoder,
    MultiScaleKVDeformableAttention,
    KVDeformableDetrTransformerEncoderLayer,
)
from ..ops.kv_deform_attn import (
    multi_scale_deformable_attn_pytorch as kv_multi_scale_deformable_attn_pytorch,
)
import matplotlib.pyplot as plt
from ....modules import DetGradCAM, visualize_features

# target_layer = "encoder.layers.5.norms.1"
target_layer = [
    "encoder.layers.5.self_attn.sampling_offsets",
    # "encoder.layers.5.self_attn.attention_weights",
]


class MultiScaleKVDeformableAttentionWithVisual(MultiScaleKVDeformableAttention):
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        identity: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        reference_points: Optional[Tensor] = None,
        spatial_shapes: Optional[Tensor] = None,
        level_start_index: Optional[Tensor] = None,
        visualize=False,
        **kwargs,
    ) -> Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if key is None:
            key = value
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
            key = key + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
            key = key.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        key = self.key_proj(key)

        if key_padding_mask is not None:
            key = key.masked_fill(key_padding_mask[..., None], 0.0)
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        key = key.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )

        query = self.query_proj(query).view(bs, num_query, self.num_heads, -1)
        query = query.permute(0, 2, 3, 1).reshape(bs * self.num_heads, -1, num_query, 1)
        # attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )
        output = kv_multi_scale_deformable_attn_pytorch(
            query, key, value, spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)
        # if visualize:
        #     self.visualize_samples(sampling_locations, attention_weights)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

    def visualize_samples(
        self, sampling_locations: Tensor, attention_weights: Tensor, shape=(640, 640)
    ):
        points = sampling_locations[0].reshape(-1, 2).detach()
        points = points.add(1).div(2)
        weights = attention_weights[0].reshape(-1).detach()
        sample_weights = points.new_zeros(shape)
        shape = points.new_tensor(shape).float()[None]
        points = points * shape
        ind = (points.gt(0) & points.lt(shape - 1)).all(-1)
        points = points[ind]
        weights = weights[ind]
        # 将浮点数折算回网格上
        left_up = points.long()
        right_bottom = left_up.add(1)
        left_bottom = left_up.clone()
        left_bottom[:, 1] = right_bottom[:, 1]
        right_up = left_up.clone()
        right_up[:, 0] = right_bottom[:, 0]
        # 分割权重 按面积比例重分
        w_left_up = (points - left_up).prod(dim=-1).mul(weights).abs()
        w_right_bottom = (points - right_bottom).prod(dim=-1).mul(weights).abs()
        w_left_bottom = (points - left_bottom).prod(dim=-1).mul(weights).abs()
        w_right_up = (points - right_up).prod(dim=-1).mul(weights).abs()
        # 去除重复
        # 计算每个点的权重
        for i in range(left_up.shape[0]):
            sample_weights[left_up[i, 1], left_up[i, 0]] += w_left_up[i]
            sample_weights[right_bottom[i, 1], right_bottom[i, 0]] += w_right_bottom[i]
            sample_weights[left_bottom[i, 1], left_bottom[i, 0]] += w_left_bottom[i]
            sample_weights[right_up[i, 1], right_up[i, 0]] += w_right_up[i]
        # 归一化
        # sample_weights = sample_weights / sample_weights.max()
        # sample_weights = sample_weights.clamp(min=0, max=1)
        torch.save(sample_weights, "sample_weights.pt")
        plt.imshow(sample_weights.cpu().numpy())
        plt.savefig("test.png")


class MultiScaleDeformableAttentionWithVisual(MultiScaleDeformableAttention):
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)

        # self.visualize(sampling_locations, attention_weights)
        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

    def visualize(self, sampling_locations: Tensor, attention_weights: Tensor):
        points = sampling_locations[0].reshape(-1, 2).detach()
        ind = points.gt(-1).all(dim=1) & (points.lt(1).all(dim=1))
        weights = attention_weights[0].reshape(-1).detach()
        px = points[ind, 0].cpu().numpy()
        py = points[ind, 1].cpu().numpy()
        weights = weights[ind].cpu().numpy()
        # 将当前的采样点画在特征图上
        plt.cla()
        plt.scatter(px, py, linewidths=0.1, s=0.1)
        plt.axis("off")
        plt.show()
        plt.savefig("sampling_locations.png", bbox_inches="tight", pad_inches=0)


class DeformableDetrWithVisualTransformerEncoderLayer(
    DeformableDetrTransformerEncoderLayer
):
    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = MultiScaleDeformableAttentionWithVisual(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)


class DeformableDetWithVisualrTransformerEncoder(DeformableDetrTransformerEncoder):
    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        super()._init_layers()
        self.layers = ModuleList(
            [
                DeformableDetrWithVisualTransformerEncoderLayer(**self.layer_cfg)
                for _ in range(self.num_layers)
            ]
        )


class DeformableDetrWithVisualTransformerDecoderLayer(
    DeformableDetrTransformerDecoderLayer
):
    """Decoder layer of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)


class DINOWithVisualTransformerDecoder(DinoTransformerDecoder):
    """Transformer Decoder of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        super()._init_layers()
        self.layers = ModuleList(
            [
                DeformableDetrTransformerDecoderLayer(**self.layer_cfg)
                for _ in range(self.num_layers - 1)
            ]
            + [DeformableDetrWithVisualTransformerDecoderLayer(**self.layer_cfg)]
        )


class KVDeformableDetrWithVisualTransformerEncoderLayer(
    KVDeformableDetrTransformerEncoderLayer
):
    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = MultiScaleKVDeformableAttentionWithVisual(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)


class KVDeformableDetWithVisualrTransformerEncoder(KVDeformableDetrTransformerEncoder):
    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        super()._init_layers()
        self.layers = ModuleList(
            [
                KVDeformableDetrWithVisualTransformerEncoderLayer(**self.layer_cfg)
                for _ in range(self.num_layers)
            ]
        )

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key_padding_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        **kwargs,
    ) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (bs, num_queries, dim)
        """
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device
        )
        visualize = False
        for lid, layer in enumerate(self.layers):
            if lid == self.num_layers - 1:
                visualize = True
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                visualize=visualize,
                **kwargs,
            )
        return query


def visualize_points(batch_data_samples, sampling_locations: Tensor):
    image = plt.imread(batch_data_samples[0].img_path)
    # sampling_locations: B x num_queries x num_heads x num_levels x num_samples x 2
    # attention_weights: B x num_queries x num_heads x num_levels x num_samples
    # start = [175, 415]
    # end = [340, 580]
    start = [0, 0]
    end = image.shape[:2]
    # 单画具有最大尺度的特征图，取其中一个头的第0尺度
    points = sampling_locations[0].reshape(-1, 2)
    ind = points.gt(0).all(dim=1) & (points.lt(1).all(dim=1))
    # 将当前的采样点画在特征图上
    plt.cla()

    plt.imshow(image[start[1] : end[1], start[0] : end[0]])
    ind = (
        ind
        & points[..., 0].gt(start[0] / image.shape[1])
        & points[..., 0].lt(end[0] / image.shape[1])
        & points[..., 1].gt(start[1] / image.shape[0])
        & points[..., 1].lt(end[1] / image.shape[0])
    )
    # 若 points 已归一化到[0,1]，需映射到像素坐标
    h, w = image.shape[:2]
    px = (points[ind, 0].cpu().numpy() * w - start[0]).astype(int)
    py = (points[ind, 1].cpu().numpy() * h - start[1]).astype(int)
    plt.scatter(px, py, c="y", linewidths=1, s=2, marker="x")
    plt.axis("off")
    plt.show()
    plt.savefig("sampling_locations.png", bbox_inches="tight", pad_inches=0)


@MODELS.register_module()
class DINOWithVisual(DINO):
    """DINO with NMS for Key Value Deformable Attention.

    Args:
        *args: Arguments for DINOWithNMS.
        **kwargs: Keyword arguments for DINOWithNMS.
    """

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = DeformableDetWithVisualrTransformerEncoder(**self.encoder)
        self.decoder = DINOWithVisualTransformerDecoder(**self.decoder)
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
            Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        cam = DetGradCAM(self, target_layer=target_layer)
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        spatial_shapes = head_inputs_dict.pop("spatial_shapes")
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples
        )
        loss = losses["loss_cls"]
        heat_map = cam(loss)  # 取第一个batch的第一个
        visualize_features(heat_map[0][0], spatial_shapes, batch_data_samples)
        return {"loss": batch_inputs.new_zeros(1).requires_grad_()}

    def forward_decoder(
        self,
        query: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        dn_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            **kwargs,
        )

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += (
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0
            )

        decoder_outputs_dict = dict(
            hidden_states=inter_states,
            references=list(references),
            spatial_shapes=spatial_shapes,
        )
        return decoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
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
        bs, _, c = memory.shape
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
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1
        )[1]
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

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
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

        visualize_points(batch_data_samples, reference_points[..., :2])
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


from .nms_dino import DINOWithNMSAlign, DINOWithNMS


@MODELS.register_module()
class SailenceDINOWithVisual(DINOWithNMS):
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = DeformableDetWithVisualrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)
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

    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        cam = DetGradCAM(self, target_layer=target_layer)
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        spatial_shapes = head_inputs_dict.pop("spatial_shapes")
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples
        )
        loss = losses["loss_cls"]
        heat_map = cam(loss)[0][0]  # 取第一个batch的第一个
        visualize_features(heat_map, spatial_shapes, batch_data_samples)
        return {"loss": batch_inputs.new_zeros(1).requires_grad_()}

    def forward_decoder(
        self,
        query: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        dn_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            **kwargs,
        )

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += (
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0
            )

        decoder_outputs_dict = dict(
            hidden_states=inter_states,
            references=list(references),
            spatial_shapes=spatial_shapes,
        )
        return decoder_outputs_dict

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

        visualize_points(batch_data_samples, reference_points[..., :2])
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


@MODELS.register_module()
class DINOWithNMSAlignWithVisual(DINOWithNMSAlign):
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = DeformableDetWithVisualrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)
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

    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        cam = DetGradCAM(self, target_layer=target_layer)
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        spatial_shapes = head_inputs_dict.pop("spatial_shapes")
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples
        )
        loss = losses["loss_cls"]
        heat_map = cam(loss)[0][0]  # 取第一个batch的第一个
        visualize_features(heat_map, spatial_shapes, batch_data_samples)
        return {"loss": batch_inputs.new_zeros(1).requires_grad_()}

    def forward_decoder(
        self,
        query: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        dn_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            **kwargs,
        )

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += (
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0
            )

        decoder_outputs_dict = dict(
            hidden_states=inter_states,
            references=list(references),
            spatial_shapes=spatial_shapes,
        )
        return decoder_outputs_dict

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
            output_memory, spatial_shapes
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

        visualize_points(batch_data_samples, reference_points[..., :2])
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
                topk_indices=topk_indices,
                topk_indices_nms=topk_indices_nms,
                dn_meta=dn_meta,
            )
            if self.training
            else dict()
        )
        return decoder_inputs_dict, head_inputs_dict


@MODELS.register_module()
class KVDINOWithVisual(DINOWithVisual):
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = KVDeformableDetWithVisualrTransformerEncoder(**self.encoder)
        self.decoder = DINOWithVisualTransformerDecoder(**self.decoder)
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
            Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)


@MODELS.register_module()
class KVDINOWithNMSAlignWithVisual(DINOWithNMSAlignWithVisual):
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = KVDeformableDetWithVisualrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)
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
