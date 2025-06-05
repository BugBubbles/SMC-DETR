from mmdet.models import (
    DeformableDETR,
    CdnQueryGenerator,
    DeformableDetrTransformerEncoder,
    DinoTransformerDecoder,
    SinePositionalEncoding,
)
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType, InstanceList
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.init import normal_
from mmcv.ops import MultiScaleDeformableAttention


@MODELS.register_module()
class DenseDeformableDetrTransformerEncoder(DeformableDetrTransformerEncoder):
    def __init__(self, *args, layer_indices=[-1], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_indices = layer_indices

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
        queries = []
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                **kwargs,
            )
            queries.append(query)
        return tuple(queries[i] for i in self.layer_indices)


@MODELS.register_module()
class DEMO(DeformableDETR):
    def __init__(self, *args, dn_cfg: OptConfigType = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.as_two_stage, "as_two_stage must be True for DINO"
        assert self.with_box_refine, "with_box_refine must be True for DINO"

        if dn_cfg is not None:
            assert (
                "num_classes" not in dn_cfg
                and "num_queries" not in dn_cfg
                and "hidden_dim" not in dn_cfg
            ), (
                "The three keyword args `num_classes`, `embed_dims`, and "
                "`num_matching_queries` are set in `detector.__init__()`, "
                "users should not set them in `dn_cfg` config."
            )
            dn_cfg["num_classes"] = self.bbox_head.num_classes
            dn_cfg["embed_dims"] = self.embed_dims
            dn_cfg["num_matching_queries"] = self.num_queries
            self.dn_query_generator = CdnQueryGenerator(**dn_cfg)
        self.dn_cfg = dn_cfg

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        if hasattr(self.train_cfg, "layer_indices"):
            self.encoder.update(layer_indices=self.train_cfg["layer_indices"])
        self.encoder = DenseDeformableDetrTransformerEncoder(**self.encoder)
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
        nn.init.xavier_uniform_(self.query_embedding.weight)
        normal_(self.level_embed)

    def predict_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        batch_img_metas: Tuple[Dict],
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
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]

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

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples
        )

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples
        )
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def pre_decoder(
        self,
        memory: Tuple[Tensor],
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
        bs, _, c = memory[0].shape
        num_layers = len(memory)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )
        output_memory = torch.stack(output_memory, 1)
        enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](
            output_memory
        )
        enc_outputs_coord_unact = self.bbox_head.reg_branches[self.decoder.num_layers](
            output_memory
        ) + output_proposals.unsqueeze(1)

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0],  # class only
            k=self.num_queries,
            dim=-1,
        )[1]
        topk_score = torch.gather(
            enc_outputs_class,
            2,
            topk_indices.unsqueeze(-1).repeat(1, 1, 1, cls_out_features),
        )
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 2, topk_indices.unsqueeze(-1).repeat(1, 1, 1, 4)
        )
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight.repeat(bs, num_layers, 1, 1)
        if self.training and self.dn_cfg is not None:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(
                batch_data_samples
            )
            dn_meta.update(num_layers=num_layers)
            # only copy
            ## TODO imporvements: each layers have its own noise query
            dn_label_query = dn_label_query.unsqueeze(1).repeat(1, num_layers, 1, 1)
            dn_bbox_query = dn_bbox_query.unsqueeze(1).repeat(1, num_layers, 1, 1)
            query = torch.cat([dn_label_query, query], dim=2)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=2)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=torch.stack(memory, 1),
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

    def gen_encoder_output_proposals(
        self, memory_list: Tuple[Tensor], memory_mask: Tensor, spatial_shapes: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Generate proposals from encoded memory. The function will only be
        used when `as_two_stage` is `True`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).

        Returns:
            tuple: A tuple of transformed memory and proposals.

            - output_memory (Tensor): The transformed memory for obtaining
              top-k proposals, has shape (bs, num_feat_points, dim).
            - output_proposals (Tensor): The inverse-normalized proposal, has
              shape (batch_size, num_keys, 4) with the last dimension arranged
              as (cx, cy, w, h).
        """
        memory = memory_list[0]
        bs = memory.size(0)
        proposals = []
        _cur = 0  # start index in the sequence of the current level
        for lvl, HW in enumerate(spatial_shapes):
            H, W = HW

            if memory_mask is not None:
                mask_flatten_ = memory_mask[:, _cur : (_cur + H * W)].view(bs, H, W, 1)
                valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1).unsqueeze(-1)
                valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1).unsqueeze(-1)
                scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
            else:
                if not isinstance(HW, torch.Tensor):
                    HW = memory.new_tensor(HW)
                scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
            proposals.append(proposal)
            _cur += H * W
        output_proposals = torch.cat(proposals, 1)
        # do not use `all` to make it exportable to onnx
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).sum(-1, keepdim=True) == output_proposals.shape[-1]
        # inverse_sigmoid
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_mask.unsqueeze(-1), float("inf")
            )
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf")
        )

        output_memory = list(memory_list)
        if memory_mask is not None:
            # use map operation to iterate only once
            output_memory = map(
                lambda x: x.masked_fill(memory_mask.unsqueeze(-1), float(0)),
                memory_list,
            )
        output_memory = map(
            lambda x: x.masked_fill(~output_proposals_valid, float(0)), output_memory
        )
        output_memory = map(self.memory_trans_fc, output_memory)
        output_memory = map(self.memory_trans_norm, output_memory)
        # [bs, sum(hw), 2]
        return list(output_memory), output_proposals

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
        bs, n, c, d = query.size()
        inter_states, references = self.decoder(
            query=query.flatten(0, 1),
            value=memory.flatten(0, 1),
            key_padding_mask=self.repeat_by_layers(memory_mask, n),
            self_attn_mask=dn_mask,
            reference_points=reference_points.flatten(0, 1),
            spatial_shapes=spatial_shapes,
            level_start_index=self.repeat_by_layers(level_start_index, n),
            valid_ratios=self.repeat_by_layers(valid_ratios, n),
            reg_branches=self.bbox_head.reg_branches,
            **kwargs,
        )

        if len(query) == self.num_queries and self.dn_cfg is not None:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += (
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0
            )

        decoder_outputs_dict = dict(
            hidden_states=inter_states.view(-1, bs, n, c, d),
            references=list(references.view(-1, bs, n, c, 4)),
        )
        return decoder_outputs_dict

    def repeat_by_layers(self, tensor: Tensor, num_layers: int):
        return (
            (
                tensor.unsqueeze(1)
                .repeat(1, num_layers, *(1,) * (len(tensor.size()) - 1))
                .flatten(0, 1)
            )
            if tensor is not None
            else None
        )


from .ops.kv_deform_attn import KVDeformableDetrTransformerEncoder
from mmdet.models import DINO


@MODELS.register_module()
class DINOWithKV(DINO):
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        if hasattr(self.train_cfg, "layer_indices"):
            self.encoder.update(layer_indices=self.train_cfg["layer_indices"])
        self.encoder = KVDeformableDetrTransformerEncoder(**self.encoder)
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


@MODELS.register_module()
class DINOWithoutSE(DINO):
    def forward_encoder(
        self,
        feat: Tensor,
        feat_mask: Tensor,
        feat_pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
    ) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat,
            query_pos=feat_pos.mul(0),
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        encoder_outputs_dict = dict(
            memory=memory, memory_mask=feat_mask, spatial_shapes=spatial_shapes
        )
        return encoder_outputs_dict
