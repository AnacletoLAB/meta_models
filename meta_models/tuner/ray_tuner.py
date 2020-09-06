"""Class implementing abstract RayTuner."""
from ray import tune
from .tuner import Tuner


class RayTuner(Tuner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)