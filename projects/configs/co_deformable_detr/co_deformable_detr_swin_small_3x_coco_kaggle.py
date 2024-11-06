_base_ = [
    'co_deformable_detr_r50_1x_coco_kaggle.py'
]
pretrained = '/kaggle/CoDETR/data/pretrained_models/co_deformable_detr_swin_small_3x_coco.pth'
# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformerV1',
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        out_indices=(1, 2, 3),
        window_size=7,
        ape=False,
        drop_path_rate=0.4,
        patch_norm=True,
        use_checkpoint=False,
        pretrained=pretrained),
    neck=dict(in_channels=[96*2, 96*4, 96*8]))

# optimizer
optimizer = dict(weight_decay=0.05)
lr_config = dict(policy='step', step=[30])
runner = dict(type='EpochBasedRunner', max_epochs=36)