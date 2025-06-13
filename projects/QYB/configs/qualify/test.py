# QYB比赛配置
_base_ = "./val.py"

# === 训练数据配置 ===

# 测试集dataloader batch size
test_batch_size = 1
# 测试集dataloader 工作进程数
test_num_workers = 2
# 测试集dataloader 是否复用工作进程
test_persistent_workers = True
# 比赛测试集标注文件绝对路径, 注意, 测试集是没有标注的, 里面只有图片信息
abs_test_ann_file = "/data3/QYB/data/RemoteSensing/test/test_images_info.json"


# 测试集数据增强配置
test_pipeline = [
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
test_dataset = dict(
    # 数据集类型
    type=_base_.dataset_type,
    # 使用比赛数据集的元信息
    metainfo=_base_.metainfo,
    # 比赛数据集根目录
    data_root=_base_.data_root,
    # 比赛验证集标注文件路径
    ann_file="test/test_images_info.json",
    # 比赛验证集图片路径前缀
    data_prefix=dict(img="test/images/"),
    # 验证集数据增强
    pipeline=test_pipeline,
)

# 测试集采样器设置
test_sampler = dict(
    _scope_="mmdet",
    shuffle=False,
    type="DefaultSampler",
)

# 测试集dataloader配置
test_dataloader = dict(
    # 数据集配置
    dataset=test_dataset,
    # 批次大小
    batch_size=test_batch_size,
    # 工作进程数
    num_workers=test_num_workers,
    # 是否复用工作进程
    persistent_workers=test_persistent_workers,
)

# 验证循环配置
test_cfg = dict(
    _scope_="mmdet",
    type="TestLoop",
)

# 验证阶段测试器配置
test_evaluator = dict(
    _scope_="mmdet",
    ann_file=abs_test_ann_file,
    format_only=True,
    outfile_prefix="mmdetection-results",
    metric=("segm"),
    type="CocoMetric",
)
