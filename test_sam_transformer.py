import torch

import sys
sys.path.append('/home/tong/code/vl_tracking/mmdetection/')
from mmtrack.models.backbones import VisionTransformer
from mmtrack.models.track_heads import TwoWayTransformer

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