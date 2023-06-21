import torch
import sys
sys.path.append('/home/tong/code/vl_tracking/mmdetection/')
from mmtrack.models.backbones import VisionTransformer

vit = VisionTransformer(
            image_size=224,
            patch_size=16,
            width=768,
            layers=12,
            heads=12,
            mlp_ratio=4.0, 
            attentional_pool=False)

image = torch.randn(2, 3, 288, 288)
tem_img = torch.randn(2, 3, 128, 128)

feat = vit(image)
tem_feat = vit(tem_img)

print(feat.shape, tem_feat.shape)