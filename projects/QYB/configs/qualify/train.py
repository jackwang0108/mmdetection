# QYB比赛配置
_base_ = ["./data.py"]

# === 训练数据配置 ===

# 训练epoch数
train_epochs = 30

# 比赛训练集标注文件路径
train_ann_file = "train/split_train_annotations.json"

# 训练集dataloader batch size
train_batch_size = 1
# 训练集dataloader 工作进程数
train_num_workers = 2
# 训练集dataloader 是否复用工作进程
train_persistent_workers = True

# 训练集数据增强配置
train_pipeline = [
    # 从文件加载图片
    dict(type="LoadImageFromFile"),
    # 加载标注信息, 包括边界框和掩码
    dict(
        type="LoadAnnotations",
        with_bbox=True,
        with_mask=True,
    ),
    # 随机翻转
    dict(type="RandomFlip", prob=0.5),
    # 随机选择增强方式
    dict(
        type="RandomChoice",
        transforms=[
            # 增强方式1: 整张图像缩放到指定尺寸
            [
                # 随机选择一个尺寸缩放
                dict(
                    type="RandomChoiceResize",
                    keep_ratio=True,
                    # fmt: off
                    scales=[
                        (480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333),
                        (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333),
                    ],
                    # fmt: on
                ),
            ],
            # 增强方式2: 针对小物体缩放
            [
                #  大大滴放大图像
                dict(
                    type="RandomChoiceResize",
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True,
                ),
                # 随机裁剪一个小区域, 希望会有小物体
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                # 裁剪后的部分图像再次缩放到指定尺寸
                dict(
                    type="RandomChoiceResize",
                    keep_ratio=True,
                    # fmt: off
                    scales=[
                        (480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333),
                        (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333),
                    ],
                    # fmt: on
                ),
            ],
        ],
    ),
    # 色彩增强
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=18,
    ),
    # 打包数据为mmdetection的标准格式
    dict(type="PackDetInputs"),
]

# 训练集配置
train_dataset = dict(
    # 删除原有配置
    # 数据集类型
    type=_base_.dataset_type,
    # 使用比赛数据集的元信息
    metainfo=_base_.metainfo,
    # 比赛数据集根目录
    data_root=_base_.data_root,
    # 比赛训练集标注文件路径
    ann_file=train_ann_file,
    # 比赛训练集图片路径前缀
    data_prefix=dict(img="train/images/"),
    # 训练集数据增强
    pipeline=train_pipeline,
)

# 训练集采样器配置
train_sampler = dict(
    _scope_="mmdet",
    shuffle=True,
    type="DefaultSampler",
)

# 训练集dataloader配置
train_dataloader = dict(
    # 数据集配置
    dataset=train_dataset,
    # 批次大小
    batch_size=train_batch_size,
    # 工作进程数
    num_workers=train_num_workers,
    # 是否复用工作进程
    persistent_workers=train_persistent_workers,
    # 采样器
    sampler=train_sampler,
)

# 训练循环配置
train_cfg = dict(
    type="EpochBasedTrainLoop",
    val_interval=1,
    max_epochs=train_epochs,
)
