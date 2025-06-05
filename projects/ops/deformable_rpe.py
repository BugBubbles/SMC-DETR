# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import ModuleList
from torch import Tensor, nn

from mmdet.models import DetrTransformerDecoderLayer, DinoTransformerDecoder
from mmdet.registry import MODELS
from mmdet.models.layers import MLP, coordinate_to_encoding, inverse_sigmoid
from typing import Tuple, Union
import numpy as np


class GlobalCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        batch_first=True,
        qkv_bias=True,
        qk_scale=None,
        dropout=0.0,
    ):
        super().__init__()
        assert batch_first, "batch_first should be True"
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.k = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.v = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query,
        key,
        value,
        rpe,
        spatial_shapes,
        key_padding_mask=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        assert spatial_shapes.size(0) == 1, "This is designed for single-scale decoder."

        B_, N, C = key.shape
        k = self.k(key).add(key_pos) if key_pos is not None else self.k(key)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        B_, N, C = value.shape
        v = (
            self.v(value)
            .reshape(B_, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        B_, N, C = query.shape
        q = self.q(query).add(query_pos) if query_pos is not None else self.q(query)
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1) * self.scale + rpe
        if key_padding_mask is not None:
            attn += key_padding_mask[:, None, None] * -100

        fmin, fmax = torch.finfo(attn.dtype).min, torch.finfo(attn.dtype).max
        torch.clip_(attn, min=fmin, max=fmax)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RPBTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Decoder layer of Deformable DETR."""

    def __init__(
        self,
        *args,
        rpe_hidden_dim=256,
        **kwargs,
    ):
        self.rpe_hidden_dim = rpe_hidden_dim
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = GlobalCrossAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
        self.cpb_mlp1 = self.build_cpb_mlp(
            2, self.rpe_hidden_dim, self.cross_attn.num_heads
        )
        self.cpb_mlp2 = self.build_cpb_mlp(
            2, self.rpe_hidden_dim, self.cross_attn.num_heads
        )

    def build_cpb_mlp(self, in_dim, hidden_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        delta_pos: Tuple[Tensor] = None,
        query_pos: Tensor = None,
        key_pos: Tensor = None,
        self_attn_mask: Tensor = None,
        cross_attn_mask: Tensor = None,
        key_padding_mask: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        if delta_pos is not None:
            rpe_x, rpe_y = self.cpb_mlp1(delta_pos[0]), self.cpb_mlp2(
                delta_pos[1]
            )  # B, nQ, w/h, nheads
            rpe = (
                (rpe_x[:, :, None] + rpe_y[:, :, :, None])
                .flatten(2, 3)
                .permute(0, 3, 1, 2)
            )  # B, nQ, h, w, nheads ->  B, nQ, h*w, nheads
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs,
        )
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            rpe=rpe,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query


@MODELS.register_module()
class RPBTransformerDecoder(DinoTransformerDecoder):
    def __init__(
        self, *args, rpe_type="linear", feature_stride=16, reparam=False, **kwargs
    ):
        self.rpe_type = rpe_type
        self.feature_stride = feature_stride
        self.reparam = reparam
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList(
            [
                RPBTransformerDecoderLayer(**self.layer_cfg)
                for _ in range(self.num_layers)
            ]
        )
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError("There is not post_norm in " f"{self._get_name()}")
        self.ref_point_head = MLP(
            self.embed_dims * 2, self.embed_dims, self.embed_dims, 2
        )
        self.norm = nn.LayerNorm(self.embed_dims)

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
        batch_input_shape: Tuple[int] = None,
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
        for lid, layer in enumerate(self.layers):
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

            delta_pos = self.relative_position_bias(
                spatial_shapes, reference_points, batch_input_shape
            )

            query = layer(
                query,
                value,
                query_pos=query_pos,
                value=value,
                delta_pos=delta_pos,
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
                new_reference_points = tmp + inverse_sigmoid(reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return query, reference_points

    def relative_position_bias(
        self, spatial_shapes, reference_points, batch_input_shape
    ):
        h, w = spatial_shapes[0]
        batch_input_shape = spatial_shapes.new_tensor([*batch_input_shape] * 2)

        ref_pts = torch.cat(
            [
                reference_points[:, :, None, :2] - reference_points[:, :, None, 2:] / 2,
                reference_points[:, :, None, :2] + reference_points[:, :, None, 2:] / 2,
            ],
            dim=-1,
        )  # B, nQ, 1, 4
        if not self.reparam:
            ref_pts *= batch_input_shape
        delta_x = (
            torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=w.device)[
                None, None, :, None
            ]
            * batch_input_shape[0]
            / w
        )  # 1, 1, w, 1
        delta_y = (
            torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=h.device)[
                None, None, :, None
            ]
            * batch_input_shape[1]
            / h
        )  # 1, 1, h, 1

        if self.rpe_type == "abs_log8":
            delta_x = ref_pts[..., 0::2] - delta_x  # B, nQ, w, 2
            delta_y = ref_pts[..., 1::2] - delta_y  # B, nQ, h, 2
            delta_x = (
                torch.sign(delta_x) * torch.log2(torch.abs(delta_x) + 1.0) / np.log2(8)
            )
            delta_y = (
                torch.sign(delta_y) * torch.log2(torch.abs(delta_y) + 1.0) / np.log2(8)
            )
        elif self.rpe_type == "linear":
            delta_x = ref_pts[..., 0::2] - delta_x  # B, nQ, w, 2
            delta_y = ref_pts[..., 1::2] - delta_y  # B, nQ, h, 2
        else:
            raise NotImplementedError
        return delta_x, delta_y
