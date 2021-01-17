"""Submodule implementing some patch that are currently needed for Ray Tune."""
import os
import sys


def patch_global_checkpoints_interval():
    """
    Currently, for some reason, Bayesian optimization slows down significantly
    when not setting the TUNE_GLOBAL_CHECKPOINT_S variable in such a way to
    avoid in the first place the checkpoint.

    The authors at Ray have been notified of this issue.

    References
    -------------------
    https://discuss.ray.io/t/extremely-slow-bo-after-random-sampling-ends/452
    """
    os.environ.setdefault("TUNE_GLOBAL_CHECKPOINT_S", str(sys.maxsize))
