# QYB 比赛配置

# === 数据集配置 ===
# 数据集类型, 比赛数据集符合COCO格式
dataset_type = "CocoDataset"

# 数据集根目录
data_root = "/data3/QYB/data/RemoteSensing/"

# 数据集类别数
num_classes = 8

# 数据集元信息
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
