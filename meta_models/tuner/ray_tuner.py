"""Class implementing abstract RayTuner."""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from ray import tune
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.schedulers import TrialScheduler
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.suggest import Searcher
from multiprocessing import cpu_count

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

    def _build_search_alg(self, space: Dict, random_search_steps: int) -> Searcher:
        """Return the tuner search algorithm.

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
            "Method _build_search_alg must be implemented in child class."
        )

    def _build_sheduler(self, max_t: int) -> TrialScheduler:
        """Return the trial scheduler.

        Parameters
        -------------------
        max_t: int,
            Maximum number of steps.

        Returns
        -------------------
        Search algorithm.
        """
        raise NotImplementedError(
            "Method _build_sheduler must be implemented in child class."
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
        verbose: int = 1,
        total_threads: int = None,
        optimizer: str = "nadam",
        loss: str = "binary_crossentropy",
    ) -> pd.DataFrame:
        """Execute tuning of dataframe."""
        if total_threads is None:
            total_threads = cpu_count()
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
            scheduler=self._build_sheduler(epochs),
            resources_per_trial={
                "cpu": cpu_count()/total_threads,
                "gpu": get_minimum_gpu_rate_per_trial(
                    total_threads
                )
            },
            num_samples=num_samples,
            stop=TrialPlateauStopper(
                metric=self._metric
            ),
            fail_fast=True,
            verbose=verbose
        ).dataframe()
