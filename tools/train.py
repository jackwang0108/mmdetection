# Copyright (c) OpenMMLab. All rights reserved.
import os
import argparse
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner


def is_debugging() -> bool:
    import sys

    return sys.gettrace() is not None


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="enable automatic-mixed-precision training",
    )
    parser.add_argument(
        "--auto-scale-lr", action="store_true", help="enable automatically scaling LR."
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        const="auto",
        help="If specify checkpoint path, resume from it, while if not "
        "specify, try to auto resume from the latest checkpoint "
        "in the work directory.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    # * 设置动态图优化占用内存, 这一行代码严重拖慢运行速度, 而在debug模式下不需要到动态图编译器进行优化, 所以没必要
    if not is_debugging():
        from mmdet.utils import setup_cache_size_limit_of_dynamo

        setup_cache_size_limit_of_dynamo()

    # * 从命令行参数中指定的配置文件路径中获取配置信息
    #   1. 读取配置文件路径的原始文本, 使用 ast (抽象语法树) 解析配置文件中的字段
    #       例如: configs/mask-rcnn_r50-caffe_fpn_ms-poly-1x_qyb.py
    #   2. 首先解析配置文件中的_base_字段, 而后加载基础配置文件中的设置
    #       例如: ../mmdetection/configs/mask_rcnn/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco.py
    #   3. 用当前配置文件中的设置覆盖基础配置文件中的设置
    #   4. 加载环境变量, 例如: CUDA_VISIBLE_DEVICES
    #   5. 返回最终的 Config 对象
    # 其中 1-4 步 是递归的, 因为基础配置文件可能还会引用其他基础配置文件
    print("Parsing config file")
    cfg = Config.fromfile(args.config)
    print("Config file parsed successfully")

    cfg.launcher = args.launcher

    # * 用命令行参数中指定的设置覆盖配置文件中的设置
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # * 设置checkpoints路径
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    # * 设置单精度FP16和双精度FP32混合训练
    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.loss_scale = "dynamic"

    # * 设置学习率自动缩放
    # enable automatically scaling LR
    if args.auto_scale_lr:
        if (
            "auto_scale_lr" in cfg
            and "enable" in cfg.auto_scale_lr
            and "base_batch_size" in cfg.auto_scale_lr
        ):
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError(
                'Can not find "auto_scale_lr" or '
                '"auto_scale_lr.enable" or '
                '"auto_scale_lr.base_batch_size" in your'
                " configuration file."
            )

    # * 设置断点续训
    # resume is determined in this priority: resume from > auto_resume
    if args.resume == "auto":
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # * 构建运行器, 如果在配置文件中没有指定运行器类型, 就用默认的运行器, 否则构建指定类型的运行器
    print("Building runner from config")
    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        # Note: Model, Dataloader, Hooks, Logger都是在这里构建的
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    print("Starting training")
    runner.train()


if __name__ == "__main__":
    main()
