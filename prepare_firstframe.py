import torch
import cv2
from torch import Tensor
import torch.nn.functional as F

import math
from copy import deepcopy
from typing import List, Optional, Tuple, Union
import glob
import numpy as np
import os

def get_cropped_img(img: Tensor, target_bbox: Tensor,
                search_area_factor: float,
                output_size: float) -> Union[Tensor, float, Tensor]:
    """ Crop Image
    Only used during testing
    This function mainly contains two steps:
    1. Crop `img` based on target_bbox and search_area_factor. If the
    cropped image/mask is out of boundary of `img`, use 0 to pad.
    2. Resize the cropped image/mask to `output_size`.

    args:
        img (Tensor): of shape (1, C, H, W)
        target_bbox (Tensor): in [cx, cy, w, h] format
        search_area_factor (float): Ratio of crop size to target size.
        output_size (float): the size of output cropped image
            (always square).
    returns:
        img_crop_padded (Tensor): of shape (1, C, output_size, output_size)
        resize_factor (float): the ratio of original image scale to cropped
            image scale.
        pdding_mask (Tensor): the padding mask caused by cropping. It's
            of shape (1, output_size, output_size).
    """
    cx, cy, w, h = target_bbox.split((1, 1, 1, 1), dim=-1)

    img_h, img_w = img.shape[2:]
    # 1. Crop image
    # 1.1 calculate crop size and pad size
    crop_size = math.ceil(math.sqrt(w * h) * search_area_factor)
    if crop_size < 1:
        raise Exception('Too small bounding box.')

    x1 = torch.round(cx - crop_size * 0.5).long()
    x2 = x1 + crop_size
    y1 = torch.round(cy - crop_size * 0.5).long()
    y2 = y1 + crop_size

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - img_w + 1, 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - img_h + 1, 0)

    # 1.2 crop image
    img_crop = img[..., y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # 1.3 pad image
    img_crop_padded = F.pad(
        img_crop,
        pad=(x1_pad, x2_pad, y1_pad, y2_pad),
        mode='constant',
        value=0)
    # 1.4 generate padding mask
    _, _, img_h, img_w = img_crop_padded.shape
    end_x = None if x2_pad == 0 else -x2_pad
    end_y = None if y2_pad == 0 else -y2_pad
    padding_mask = torch.ones((img_h, img_w),
                                dtype=torch.float32,
                                device=img.device)
    padding_mask[y1_pad:end_y, x1_pad:end_x] = 0.

    # 2. Resize cropped image and padding mask
    resize_factor = output_size / crop_size
    img_crop_padded = F.interpolate(
        img_crop_padded, (output_size, output_size),
        mode='bilinear',
        align_corners=False)

    padding_mask = F.interpolate(
        padding_mask[None, None], (output_size, output_size),
        mode='bilinear',
        align_corners=False).squeeze(dim=0).type(torch.bool)

    return img_crop_padded, resize_factor, padding_mask

dataset_path = '/ocean/projects/ele220002p/tongshen/dataset/lasot/LaSOTBenchmark/'
output_folder = '//ocean/projects/ele220002p/tongshen/code/vl_tracking/first_frames/lasot/'
data_folders = glob.glob(dataset_path + '/*/')

for cate_folder in data_folders:
    folders = glob.glob(cate_folder + '/*/')
    
    for ff in folders:
        
        img = cv2.imread(ff + '/img/00000001.jpg')
        gt_bbox = open(ff + "groundtruth.txt").readline().rstrip()
        gt_bbox = np.array(list(map(int, gt_bbox.split(','))))
        
        bbox_xyxy = gt_bbox.copy()
        bbox_xyxy[2:] += bbox_xyxy[:2]
        
        bbox_cxcywh = gt_bbox.copy()
        bbox_cxcywh[:2] += bbox_cxcywh[2:]//2
        bbox_cxcywh = np.array(bbox_cxcywh)
        
        img = cv2.rectangle(img, tuple(bbox_xyxy[:2]), tuple(bbox_xyxy[2:]), (255, 0, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        basename = ff.split('/')[-2]
        cate_name = ff.split('/')[-3]
        os.makedirs(output_folder + cate_name, exist_ok=True)
        
        cv2.imwrite(output_folder + cate_name + '/' + basename + '.png', img)
        
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        target_box = torch.from_numpy(bbox_cxcywh)
        
        search_area_factor = 2
        output_size = 224
        
        img_crop_padded, resize_factor, padding_mask = get_cropped_img(img_tensor, target_box, search_area_factor, output_size)
        img_crop = img_crop_padded[0].permute(1, 2, 0).numpy().astype(np.uint8)
        print(output_folder + cate_name + '/' + basename + '_crop.png')
        
        cv2.imwrite(output_folder + cate_name + '/' + basename + '_crop.png', img_crop)
        
        