_base_ = ['../../_base_/default_runtime.py']

randomness = dict(seed=1, deterministic=False)

# model setting
model = dict(
    type='VLTracker',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='VisionTransformer',
        image_size=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        mlp_ratio=4.0,
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    head=dict(
        type='MultiModalFusionHead',
        transformer=dict(
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
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=1.0),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=1.0),
        loss_quality=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0)
    ),
    train_cfg=dict(
        feat_size=(18, 18),
        img_size=(288, 288),
        sigma_factor=0.05,
        end_pad_if_even=True,
        gauss_label_bias=0.,
        use_gauss_density=True,
        label_density_norm=True,
        label_density_threshold=0.,
        label_density_shrink=0,
        loss_weights=dict(cls_init=0.25, cls_iter=1., cls_final=0.25)
    ),
    test_cfg=dict(
        img_sample_size=22 * 16,
        feature_stride=16,
        search_scale_factor=6,
        patch_max_scale_change=1.5,
        border_mode='inside_major',
    )
)
    
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
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadTrackAnnotations', with_instance_id=False),
            dict(type='GrayAug', prob=0.05),
            dict(type='mmdet.RandomFlip', prob=0.5, direction='horizontal')
        ]),
    dict(
        type='SeqBboxJitter',
        center_jitter_factor=[3, 3, 4.5],
        scale_jitter_factor=[0.25, 0.25, 0.5],
        crop_size_factor=[2, 2, 5]),
    dict(
        type='SeqCropLikeStark',
        crop_size_factor=[2, 2, 5],
        output_size=[128, 128, 288]),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[dict(type='BrightnessAug', jitter_range=0.2)]),
    dict(type='CheckPadMaskValidity', stride=16),
    dict(type='PackTrackInputs', ref_prefix='search', num_template_frames=2)
]

data_root = '/home/tong/dataset/'
# dataset settings
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='QuotaSampler', samples_per_epoch=60000),
    dataset=dict(
        # type='RandomSampleConcatDataset',
        # # dataset_sampling_weights=[1, 1, 1, 1],
        # dataset_sampling_weights=[1],
        # datasets=[
        #     # dict(
        #     #     type='GOT10kDataset',
        #     #     data_root=data_root,
        #     #     ann_file='GOT10k/annotations/got10k_train_vot_infos.txt',
        #     #     data_prefix=dict(img_path='GOT10k'),
        #     #     pipeline=train_pipeline,
        #     #     test_mode=False),
        #     dict(
                type='LaSOTDataset',
                data_root=data_root,
                ann_file='lasot/annotations/lasot_train_infos.txt',
                data_prefix=dict(img_path='lasot/LaSOTBenchmark'),
                pipeline=train_pipeline,
                test_mode=False),
            # dict(
            #     type='TrackingNetDataset',
            #     chunks_list=[0, 1, 2, 3],
            #     data_root=data_root,
            #     ann_file='TrackingNet/annotations/trackingnet_train_infos.txt',
            #     data_prefix=dict(img_path='TrackingNet'),
            #     pipeline=train_pipeline,
            #     test_mode=False),
            # dict(
            #     type='SOTCocoDataset',
            #     data_root=data_root,
            #     ann_file='coco/annotations/instances_train2017.json',
            #     data_prefix=dict(img_path='coco/train2017'),
            #     pipeline=train_pipeline,
            #     test_mode=False)
        )

# runner loop
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=100, val_interval=1)

# learning policy
param_scheduler = dict(type='MultiStepLR', milestones=[60, 90], gamma=0.2)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=2e-4),
    # clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_multi=0.1),
            head=dict(lr_multi=1),)
        )
    )

# checkpoint saving
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10),
    logger=dict(type='LoggerHook', interval=50))
