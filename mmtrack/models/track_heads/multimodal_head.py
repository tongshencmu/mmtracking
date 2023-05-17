import torch 
import torch.nn as nn
from typing import List, Tuple, Type

from mmtrack.registry import MODELS
from .multimodal_transformer import TwoWayAttentionBlock

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
class MultiModelHead(nn.Module):
    
    def __init__(self,
                 transformer_dim, 
                 transformer, 
                 
                 activation: Type[nn.Module] = nn.GELU,
                 ) -> None:
        super().__init__()
        
        self.transformer = transformer
        self.transformer_dim = transformer_dim
        
        self.
        
        self.prediction_token = nn.Embedding(1, transformer_dim)
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        
    def forward(self, 
                x_embeddings,
                x_pe, 
                z_embeddings, 
                z_pe,
                text_embeddings):
        
        # Concatenate z_embeddings and text_embeddings into prompt embeddings and go through transformer
        prompt_embeddings = torch.cat([z_embeddings, text_embeddings], dim=-1)
        
        
        