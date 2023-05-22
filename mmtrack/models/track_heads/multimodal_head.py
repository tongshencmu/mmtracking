# Adapted from Segment Anything 
# https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py


import torch 
import torch.nn as nn
from typing import List, Tuple, Type, Any, Optional

import numpy as np

from mmtrack.registry import MODELS
from .sam_transformer import TwoWayAttentionBlock

from mmtrack.utils import InstanceList, OptConfigType, SampleList

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@MODELS.register_module()
class MultiModelFusion(nn.Module):
    
    def __init__(self,
                 transformer, 
                 bbox_head, 
                 quality_head,
                 transformer_dim,
                 search_feat_size,
                 template_feat_size,
                 ) -> None:
        super().__init__()
        
        self.transformer = MODELS.build(transformer)
        self.transformer_dim = transformer_dim
        
        self.bbox_head = MODELS.build(bbox_head)
        self.quality_head = MODELS.build(quality_head)
        
        self.template_feat_size = template_feat_size
        self.search_feat_size = search_feat_size
        
        self.template_pe_layer = PositionEmbeddingRandom(num_pos_feats=transformer_dim)
        self.search_pe_layer = PositionEmbeddingRandom(num_pos_feats=transformer_dim)
        
        self.pred_quality_token = nn.Embedding(1, transformer_dim)
        
    def forward(self, 
                x_embeddings,
                z_embeddings, 
                text_embeddings=None):
        
        # Concatenate z_embeddings and text_embeddings into prompt embeddings and go through transformer
        prompt_pe = self.template_pe_layer([self.template_feat_size, self.template_feat_size])
        image_pe = self.search_pe_layer([self.search_feat_size, self.search_feat_size])
        
        if text_embeddings is not None:
            prompt_embeddings = torch.cat([z_embeddings, text_embeddings], dim=-1)
            prompt_pe = torch.cat([prompt_pe, text_embeddings], dim=-1)
        else:
            prompt_embeddings = z_embeddings
            
        assert x_embeddings.shape[0:2] == prompt_embeddings.shape[0:2]
        b, c, h_z, w_z = z_embeddings.shape
        _, _, h_x, w_x = x_embeddings.shape
        
        # prompt_embeddings = prompt_embeddings.permute(0, 3, 1, 2)
        prompt_embeddings = torch.cat([prompt_embeddings, self.pred_quality_token], dim=1)
        prompt_embeddings, image_embeddings = self.transformer(x_embeddings, image_pe, prompt_embeddings, prompt_pe)
        
        token_emb = prompt_embeddings[:, 0, :]
        z_emb = prompt_embeddings[:, 1:, :]
        
        # Run box head for regression
        output_coords = self.bbox_head(image_embeddings)
        
        # Generate tracking quality prediction using token embedding
        quality_pred = self.quality_head(token_emb)
        
        return output_coords, quality_pred
    
    def loss(self, inputs: List[dict], data_samples: SampleList,
             **kwargs) -> dict:
        """Compute loss.

        Args:
            inputs (list[dict(tuple(Tensor))]): The list contains the
                multi-level features and masks of template or search images.
                    - 'feat': (tuple(Tensor)), the Tensor is of shape
                        (bs, c, h//stride, w//stride).
                    - 'mask': (Tensor), of shape (bs, h, w).
                Here, `h` and `w` denote the height and width of input
                image respectively. `stride` is the stride of feature map.
            data_samples (List[:obj:`TrackDataSample`]): The Data
                Samples. It usually includes information such as `gt_instance`
                and 'metainfo'.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        outs = self(inputs)

        batch_gt_instances = []
        batch_img_metas = []
        batch_search_gt_instances = []
        for data_sample in data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            batch_search_gt_instances.append(data_sample.search_gt_instances)

        loss_inputs = outs + (batch_gt_instances, batch_search_gt_instances,
                              batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)

        return losses
        
        
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C    