# %%writefile /content/drive/MyDrive/Colab_Notebooks/data/Upstage_recycling_Object_Detection/mmdetection/configs_custum/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py
# 커스텀 모듈 임포트
custom_imports = dict(imports=['projects.CO-DETR.codetr'], allow_failed_imports=False)

# 데이터셋 설정
data_root = '/content/drive/MyDrive/Colab_Notebooks/data/Upstage_recycling_Object_Detection/recycling/'
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')
dataset_type = 'CocoDataset'
default_scope = 'mmdet'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
load_from = '/content/drive/MyDrive/Colab_Notebooks/data/Upstage_recycling_Object_Detection/mmdetection/checkpoints/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'

# 파이프라인
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [dict(type='RandomChoiceResize', scales=[(768, 768), (896, 896), (1024, 1024)], keep_ratio=True)],
            [dict(type='RandomChoiceResize', scales=[(500, 1024), (600, 1024), (700, 1024)], keep_ratio=True),
             dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
             dict(type='RandomChoiceResize', scales=[(768, 768), (896, 896), (1024, 1024)], keep_ratio=True)]
        ]),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# TTA 파이프라인
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale=s, keep_ratio=True) for s in [(1024, 1024), (896, 896), (1152, 1152)]],
            [dict(type='RandomFlip', prob=1.), dict(type='RandomFlip', prob=0.)],
            [dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction'))]
        ])
]

# 데이터로더 설정
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True, _scope_='mmdet'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline,
        classes=classes,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=None
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmdet'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/'),
        pipeline=test_pipeline,
        classes=classes,
        test_mode=True,
        backend_args=None
    )
)

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmdet'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/'),
        pipeline=test_pipeline,
        classes=classes,
        test_mode=True,
        backend_args=None
    )
)

# 모델 설정 (불필요한 키 제거)
model = dict(
    type='CoDETR',
    _scope_='mmdet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32)),
    query_head=dict(
        type='CoDINOHead',
        num_query=900,
        num_classes=10,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        mixed_selection=True,
        dn_cfg=dict(
            box_noise_scale=0.4,
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=500)),
        transformer=dict(
            type='CoDinoTransformer',
            two_stage=True,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                with_cp=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=5),
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1),
                        dict(type='MultiScaleDeformableAttention', embed_dims=256, num_heads=8, num_levels=5)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=2.5),
        loss_iou=dict(type='GIoULoss', loss_weight=1.0))
)

# TTA 모델 설정
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)
)

# 학습 설정
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)
max_epochs = 16
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop', _scope_='mmdet')
test_cfg = dict(type='TestLoop', _scope_='mmdet')
param_scheduler = [dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[8, 14], gamma=0.1)]

# 기본 훅
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmdet'),
    logger=dict(type='LoggerHook', interval=50, _scope_='mmdet'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmdet'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=1, _scope_='mmdet'),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmdet'),
    visualization=dict(type='DetVisualizationHook', _scope_='mmdet')
)
custom_hooks = [dict(type='Fp16CompressHook', priority='HIGH', _scope_='mmengine')]

# 환경 설정
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# 평가 및 시각화
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    format_only=False,
    backend_args=None,
    _scope_='mmdet'
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    format_only=False,
    backend_args=None,
    _scope_='mmdet'
)

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend', _scope_='mmdet')],
    name='visualizer',
    _scope_='mmdet'
)

# 작업 디렉토리
work_dir = '/content/drive/MyDrive/Colab_Notebooks/data/Upstage_recycling_Object_Detection/mmdetection/work_dirs'