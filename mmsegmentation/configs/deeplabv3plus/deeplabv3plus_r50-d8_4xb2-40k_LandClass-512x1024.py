_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/land_class.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_10k.py'
]
crop_size = (128, 128)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
