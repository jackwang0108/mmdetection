# QYB比赛设置

default_scope = "mmdet"

launcher = "none"


log_level = "INFO"

log_processor = dict(
    _scope_="mmdet",
    by_epoch=True,
    type="LogProcessor",
    window_size=50,
)

env_cfg = dict(
    # 内存和速度优化
    cudnn_benchmark=False,  # 是否启用cudnn加速
    # 分布式设置
    dist_cfg=dict(
        # 通信后端
        backend="nccl",
    ),
    # 工作进程
    mp_cfg=dict(
        mp_start_method="fork",
        opencv_num_threads=0,
    ),
)

vis_backends = [
    dict(_scope_="mmdet", type="LocalVisBackend"),
]

visualizer = dict(
    _scope_="mmdet",
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=vis_backends,
)
