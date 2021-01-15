"""Class implementing abstract RayTuner."""
from typing import Tuple, Dict
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.keras import TuneReportCallback
from ..utils import get_minimum_gpu_rate_per_trial
from .tuner import Tuner


class RayTuner(Tuner):

    def __init__(
        self,
        num_samples: int,
        metric: str = "val_loss",
        mode: str = "min",
        **kwargs
    ):
        super().__init__(**kwargs)
        self._num_samples = num_samples
        self._metric = metric
        self._mode = mode

    def _build_search_alg(self, space: Dict) -> "SerchAlg":
        """Tune the model."""
        raise NotImplementedError(
            "Method tune must be implemented in child class."
        )

    def tune(
        self,
        train: Tuple[np.ndarray],
        validation_data: Tuple[np.ndarray],
        epochs: int,
        batch_size: int,
        patience: int,
        min_delta: float,
        name: str,
        optimizer: str = "nadam",
        loss: str = "binary_crossentropy",
    ):
        asha_scheduler = ASHAScheduler(
            time_attr='training_iteration',
            max_t=epochs,
            grace_period=10,
            reduction_factor=3
        )
        return tune.run(
            tune.with_parameters(
                self.fit,
                train=train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                min_delta=min_delta,
                optimizer=optimizer,
                loss=loss,
                callbacks=(TuneReportCallback(),),
                subgpu_training=True,
            ),
            metric=self._metric,
            mode=self._mode,
            name=name,
            search_alg=self._build_search_alg(self._meta_model.space()),
            scheduler=asha_scheduler,
            resources_per_trial={
                "gpu": get_minimum_gpu_rate_per_trial()
            },
            num_samples=self._num_samples,
            fail_fast=True,
            verbose=1
        ).dataframe()
