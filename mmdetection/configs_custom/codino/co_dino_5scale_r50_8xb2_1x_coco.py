# 기본 설정
custom_imports = dict(imports=['projects.CO-DETR.codetr'], allow_failed_imports=False)
default_scope = 'mmdet'
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), dist_cfg=dict(backend='nccl'))
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
launcher = 'none'

# 데이터셋 설정
data_root = '/content/drive/MyDrive/Colab_Notebooks/data/Upstage_recycling_Object_Detection/recycling/'
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')
dataset_type = 'CocoDataset'

# 파이프라인
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [dict(type='RandomChoiceResize', scales=[(768, 768), (896, 896), (1024, 1024)], keep_ratio=True)],
            [
                dict(type='RandomChoiceResize', scales=[(500, 1024), (600, 1024), (700, 1024)], keep_ratio=True),
                dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                dict(type='RandomChoiceResize', scales=[(768, 768), (896, 896), (1024, 1024)], keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# 데이터로더
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=None
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/'),
        metainfo=dict(classes=classes),
        pipeline=test_pipeline,
        backend_args=None,
        test_mode=True
    )
)

test_dataloader = val_dataloader

# 모델 설정 (ResNet-50)
model = dict(
    type='CoDETR',
    use_lsj=False,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
        batch_augments=None
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5
    ),
    query_head=dict(
        type='CoDINOHead',
        num_query=900,
        num_classes=10,
        in_channels=2048,
        as_two_stage=True,
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=1.0,
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)
        ),
        transformer=dict(
            type='CoDinoTransformer',
            with_coord_feat=False,
            num_co_heads=2,
            num_feature_levels=5,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                with_cp=4,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(type='MultiScaleDeformableAttention', embed_dims=256, num_levels=5, dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')
                )
            ),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention', embed_dims=256, num_heads=8, dropout=0.0),
                        dict(type='MultiScaleDeformableAttention', embed_dims=256, num_levels=5, dropout=0.0)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                )
            )
        ),
        positional_encoding=dict(type='SinePositionalEncoding', num_feats=128, temperature=20, normalize=True),
        loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(type='AnchorGenerator', octave_base_scale=4, scales_per_octave=3, ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0., 0., 0., 0.], target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=12.0),
        loss_bbox=dict(type='L1Loss', loss_weight=12.0)
    ),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56
            ),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0., 0., 0., 0.], target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=12.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=120.0)
            )
        )
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=10,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(type='AnchorGenerator', ratios=[1.0], octave_base_scale=8, scales_per_octave=1, strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0., 0., 0., 0.], target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=12.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=24.0),
            loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=12.0)
        )
    ],
    train_cfg=[
        dict(assigner=dict(type='HungarianAssigner', match_costs=[dict(type='FocalLossCost', weight=2.0), dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'), dict(type='IoUCost', iou_mode='giou', weight=2.0)])),
        dict(
            rpn=dict(
                assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
                sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False
            ),
            rpn_proposal=dict(nms_pre=4000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
            rcnn=dict(
                assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5, match_low_quality=False, ignore_iof_thr=-1),
                sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False
            )
        ),
        dict(assigner=dict(type='ATSSAssigner', topk=9), allowed_border=-1, pos_weight=-1, debug=False)
    ],
    test_cfg=[
        dict(max_per_img=300, nms=dict(type='soft_nms', iou_threshold=0.8)),
        dict(
            rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
            rcnn=dict(score_thr=0.0, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)
        ),
        dict(nms_pre=1000, min_bbox_size=0, score_thr=0.0, nms=dict(type='nms', iou_threshold=0.6), max_per_img=100)
    ],
    eval_module='detr'
)

# 학습 설정
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)

max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[11], gamma=0.1)]

load_from = None  # ResNet-50은 사전 학습 체크포인트 별도 지정 필요 시 추가
resume = False

# 평가 및 시각화
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    backend_args=None
)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# 작업 디렉토리
work_dir = '/content/drive/MyDrive/Colab_Notebooks/data/Upstage_recycling_Object_Detection/mmdetection/work_dirs'