import torch

import sys
sys.path.append('/home/tong/code/vl_tracking/mmdetection/')
from mmtrack.models.backbones import VisionTransformer
from mmtrack.models.track_heads import TwoWayTransformer
from mmtrack.models.track_heads import MultiModalHead

sam_t = TwoWayTransformer(depth=3,
            embedding_dim=768,
            num_heads=8,
            mlp_dim=2048,)

image = torch.randn(2, 768, 18, 18)
tem_img = torch.randn(2, 64, 768)

img_pe = torch.randn(2, 768, 18, 18)
tem_img_pe = torch.randn(2, 64, 768)

q, v = sam_t(image, img_pe, tem_img, tem_img_pe)
print(q.shape, v.shape)

head = MultiModalHead(transformer=dict(
            type='TwoWayTransformer',
            depth=3,
            embedding_dim=768,
            num_heads=8,
            mlp_dim=2048,
        ),
        transformer_dim=768, 
        template_feat_size=8,
        search_feat_size=18,
        bbox_head=dict(
            type='CornerPredictorHead',
            inplanes=768,
            channel=768,
            feat_size=18,
            stride=16
        ),
)

image = torch.randn(2, 324, 768)
tem_img = torch.randn(2, 64, 768)

head(image, tem_img)