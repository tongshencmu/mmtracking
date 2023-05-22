import torch
import sys
sys.path.append('/home/tong/code/vl_tracking/mmdetection/')
from mmtrack.models.backbones import VisionTransformer

vit = VisionTransformer(
            image_size=288,
            patch_size=18,
            width=384,
            layers=12,
            heads=6,
            mlp_ratio=4.0)

image = torch.randn(2, 3, 288, 288)
tem_img = torch.randn(2, 3, 126, 126)

feat = vit(image)
tem_feat = vit(tem_img)