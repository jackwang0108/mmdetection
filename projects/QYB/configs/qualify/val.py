# QYB比赛配置
_base_ = "./train.py"

# === 训练数据配置 ===

# 比赛验证集标注文件路径
val_ann_file = "train/split_val_annotations.json"
# 比赛验证集标注文件绝对路径
abs_val_ann_file = "/data3/QYB/data/RemoteSensing/train/split_val_annotations.json"
# 验证集dataloader batch size
val_batch_size = 1
# 验证集dataloader 工作进程数
val_num_workers = 2
# 验证集dataloader 是否复用工作进程
val_persistent_workers = True


# 验证集数据增强配置
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="Resize",
        scale=(1333, 800),
        keep_ratio=True,
    ),
    dict(
        type="LoadAnnotations",
        with_bbox=True,
        with_mask=True,
    ),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]

# 验证集配置
val_dataset = dict(
    # 数据集类型
    type=_base_.dataset_type,
    # 使用比赛数据集的元信息
    metainfo=_base_.metainfo,
    # 比赛数据集根目录
    data_root=_base_.data_root,  # type: ignore
    # 比赛验证集标注文件路径
    ann_file=val_ann_file,
    # 比赛验证集图片路径前缀
    data_prefix=dict(img="train/images/"),
    # 验证集数据增强
    pipeline=val_pipeline,
)

# 验证集采样器设置
val_sampler = dict(
    _scope_="mmdet",
    shuffle=False,
    type="DefaultSampler",
)

# 验证集dataloader配置
val_dataloader = dict(
    # 数据集配置
    dataset=val_dataset,
    # 批次大小
    batch_size=val_batch_size,
    # 工作进程数
    num_workers=val_num_workers,
    # 是否复用工作进程
    persistent_workers=val_persistent_workers,
)

# 验证循环配置
val_cfg = dict(
    _scope_="mmdet",
    type="ValLoop",
)

# 验证阶段测试器配置
val_evaluator = dict(
    _scope_="mmdet",
    ann_file=abs_val_ann_file,
    format_only=False,
    metric=("segm"),
    type="CocoMetric",
)
