# lr_scheduler.py
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import detectron2.projects.deeplab.build_solver as deeplab_build
import detectron2.solver.build as solver_build


_original = solver_build.build_lr_scheduler


def patched_lr_scheduler(cfg, optimizer):
    name = cfg.SOLVER.LR_SCHEDULER_NAME

    if name == "CosineWarmRestartsNoWarmup":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=30000,
            T_mult=1,
            eta_min=1e-6,
        )
    else:
        return _original(cfg, optimizer)


solver_build.build_lr_scheduler = patched_lr_scheduler
deeplab_build.build_d2_lr_scheduler = patched_lr_scheduler