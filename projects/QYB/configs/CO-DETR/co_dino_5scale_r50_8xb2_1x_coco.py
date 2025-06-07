_base_ = "./co_dino_5scale_r50_lsj_8xb2_1x_coco.py"

metainfo = {
    "classes": (
        "storage_tank",
        "vehicle",
        "aircraft",
        "ship",
        "bridge",
        "sports_facility",
        "roundabout",
        "harbor",
    ),
    "palette": [
        (220, 20, 60),
        (119, 11, 32),
        (0, 0, 142),
        (0, 60, 100),
        (0, 80, 100),
        (166, 189, 219),
        (28, 144, 153),
        (173, 221, 142),
    ],
}


model = dict(
    use_lsj=False,
    data_preprocessor=dict(pad_mask=False, batch_augments=None),
)

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_base_.backend_args),
    dict(
        type="LoadAnnotations",
        with_bbox=True,
        with_mask=True,
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="RandomChoiceResize",
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=18,
    ),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        metainfo=metainfo,
        type=_base_.dataset_type,
        data_root=_base_.data_root,
        ann_file="train/split_train_annotations.json",
        data_prefix=dict(img="train/images/"),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=_base_.backend_args,
    ),
)

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_base_.backend_args),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(
        type="LoadAnnotations",
        with_bbox=True,
        with_mask=True,
    ),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        pipeline=test_pipeline,
    )
)
test_dataloader = val_dataloader
