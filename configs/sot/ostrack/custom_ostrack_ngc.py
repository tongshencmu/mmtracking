_base_ = ['../../_base_/default_runtime.py']

randomness = dict(seed=1, deterministic=False)

# model settings
model = dict(
    type='OSTrack',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    # backbone=dict(
    #     type='mmdet.ResNet',
    #     depth=50,
    #     num_stages=3,
    #     strides=(1, 2, 2),
    #     dilations=[1, 1, 1],
    #     out_indices=[2],
    #     frozen_stages=1,
    #     norm_eval=True,
    #     norm_cfg=dict(type='BN', requires_grad=False),
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    backbone=dict(
        type='OSTrackViT',
        image_size=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        mlp_ratio=4.0,
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='/workspace/vit-b-16-laion-2b_visual.pth')
    ),
    # neck=dict(
    #     type='mmdet.ChannelMapper',
    #     in_channels=[768],
    #     out_channels=256,
    #     kernel_size=1,
    #     act_cfg=None),
    head=dict(
        type='OSTrackHead',
        feat_sz=20,
        bbox_head=dict(
            type='CenterPredictHead',
            inplanes=768,
            channel=256,
            feat_sz=20,
            stride=16),
        loss_cls=dict(type='FocalLoss'),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=5.0),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=2.0)),
    test_cfg=dict(
        search_factor=5.0,
        search_size=320,
        template_factor=2.0,
        template_size=128,
        num_templates=2))

data_root = '/'
train_pipeline = [
    dict(
        type='DiMPSampling',
        num_search_frames=1,
        num_template_frames=2,
        max_frame_range=200),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_instance_id=False),
            dict(type='GrayAug', prob=0.05),
            dict(type='mmdet.RandomFlip', prob=0.5, direction='horizontal')
        ]),
    dict(
        type='SeqBboxJitter',
        center_jitter_factor=[0, 0, 4.5],
        scale_jitter_factor=[0, 0, 0.5],
        crop_size_factor=[2, 2, 5]),
    dict(
        type='SeqCropLikeStark',
        crop_size_factor=[2, 2, 5],
        output_size=[128, 128, 320]),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[dict(type='BrightnessAug', jitter_range=0.2)]),
    dict(type='CheckPadMaskValidity', stride=16),
    dict(type='PackTrackInputs', ref_prefix='search', num_template_frames=2)
]

# dataset settings
batch_size=48
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='QuotaSampler', samples_per_epoch=60000),
    dataset=dict(
        type='RandomSampleConcatDataset',
        dataset_sampling_weights=[1, 1, 1, 1],
        # dataset_sampling_weights=[1],
        datasets=[
            dict(
                type='GOT10kDataset',
                data_root='/tracking_data/',
                ann_file='got10k/annotations/got10k_train_infos.txt',
                data_prefix=dict(img_path='got10k'),
                pipeline=train_pipeline,
                test_mode=False),
            dict(
                type='LaSOTDataset',
                data_root='',
                ann_file='./lasot_train_infos.txt',
                data_prefix=dict(img_path='/lasot/LaSOT/LaSOT_benchmark'),
                pipeline=train_pipeline,
                test_mode=False),
            dict(
                type='TrackingNetDataset',
                chunks_list=[0, 1, 2, 3],
                data_root='/tracking_data/',
                ann_file='trackingnet/annotations/trackingnet_train_infos.txt',
                data_prefix=dict(img_path='trackingnet'),
                pipeline=train_pipeline,
                test_mode=False),
            dict(
                type='SOTCocoDataset',
                data_root='/tracking_data/',
                ann_file='coco/annotations/instances_train2017.json',
                data_prefix=dict(img_path='coco/train2017'),
                pipeline=train_pipeline,
                test_mode=False)
        ],
    ))

# runner loop
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=300, val_begin=300, val_interval=1)

# learning policy
param_scheduler = dict(type='MultiStepLR', milestones=[240], gamma=0.1)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0004, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.2, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0)))
    )

# checkpoint saving
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=4))

# auto_scale_lr = dict(enable=True, base_batch_size=batch_size)
