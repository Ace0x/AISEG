# dataset settings
dataset_type = 'LandClassDataset'
data_root = 'data/land_cover/'
crop_size = (128, 128)

# Define the training pipeline with the necessary augmentations
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),  # Ensure no zero label reduction for grayscale
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=15),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# Define the test pipeline for validation and testing
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),  # Ensure no zero label reduction for grayscale
    dict(type='PackSegInputs')
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

# Data loader for training
train_dataloader = dict(
    batch_size=32,
    num_workers=8,  # Adjusted number of workers
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline)
)

# Data loader for validation
val_dataloader = dict(
    batch_size=16,
    num_workers=4,  # Adjusted number of workers
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline)
)

# Reuse validation dataloader settings for testing
test_dataloader = val_dataloader

# Evaluators for validation and testing
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

resume_from = '../../../work_dirs/deeplabv3_r50-d8_4xb2-40k_LandClass-512x1024/iter_18000.pth'