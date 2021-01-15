"""Class implementing abstract RayTuner."""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from ray import tune
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper

from ..meta_models import MetaModel
from ..utils import get_minimum_gpu_rate_per_trial
from .tuner import Tuner


class RayTuner(Tuner):

    def __init__(
        self,
        meta_model: MetaModel,
        metric: str = "val_loss",
        mode: str = "min",
    ):
        """Create the Tuner object.

        Parameters
        --------------------------
        meta_model: MetaModel,
            The meta-model to optimize.
        metric: str = "val_loss",
            The metric to tune for.
        mode: str = "min",
            The modality to tune the metric towards.
        """
        super().__init__(
            meta_model=meta_model
        )
        self._metric = metric
        self._mode = mode

    def _build_search_alg(self, space: Dict, random_search_steps: int) -> "SerchAlg":
        """Tune the model.

        Parameters
        -------------------
        space: Dict,
            Space of hyper-parameters.
        random_search_steps: int,
            Number of random search steps.

        Returns
        -------------------
        Search algorithm.
        """
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
        num_samples: int,
        random_search_steps: int,
        cpu: int = 1,
        optimizer: str = "nadam",
        loss: str = "binary_crossentropy",
    ) -> pd.DataFrame:
        """Execute tuning of dataframe."""
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
            search_alg=self._build_search_alg(
                self._meta_model.space(),
                random_search_steps=random_search_steps
            ),
            scheduler=asha_scheduler,
            resources_per_trial={
                "cpu": cpu,
                "gpu": get_minimum_gpu_rate_per_trial()
            },
            num_samples=num_samples,
            stop=TrialPlateauStopper(
                metric=self._metric
            ),
            fail_fast=True,
            verbose=1
        ).dataframe()
