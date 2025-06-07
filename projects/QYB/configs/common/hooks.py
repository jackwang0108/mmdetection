# QYB比赛设置

default_hooks = dict(
    checkpoint=dict(
        _scope_="mmdet",
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        type="CheckpointHook",
    ),
    logger=dict(_scope_="mmdet", interval=50, type="LoggerHook"),
    param_scheduler=dict(_scope_="mmdet", type="ParamSchedulerHook"),
    sampler_seed=dict(_scope_="mmdet", type="DistSamplerSeedHook"),
    timer=dict(_scope_="mmdet", type="IterTimerHook"),
    visualization=dict(_scope_="mmdet", type="DetVisualizationHook"),
)
