"""Class implementing abstract RayTuner."""
from multiprocessing import cpu_count
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from ray import tune
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.suggest import Searcher

from ..meta_models import MetaModel
from ..utils import get_minimum_gpu_rate_per_trial
from .tuner import Tuner


class RayTuner(Tuner):

    def __init__(
        self,
        meta_model: MetaModel,
        metric: str = "val_AUPRC",
        mode: str = "max",
    ):
        """Create the Tuner object.

        Parameters
        --------------------------
        meta_model: MetaModel,
            The meta-model to optimize.
        metric: str = "val_AUPRC",
            The metric to tune for.
        mode: str = "max",
            The modality to tune the metric towards.
        """
        super().__init__(
            meta_model=meta_model
        )
        self._metric = metric
        self._mode = mode
        self._analysis = None

    def _parse_space(self) -> Dict:
        """Return the training space adapted for the considered algorithm.

        Returns
        -------------------
        Search algorithm.
        """
        raise NotImplementedError(
            "Method _build_search_alg must be implemented in child class."
        )

    def _build_search_alg(self, **kwargs: Dict) -> Searcher:
        """Return the tuner search algorithm.

        Parameters
        -------------------
        random_search_steps: int,
            Number of random search steps.

        Returns
        -------------------
        Search algorithm.
        """
        raise NotImplementedError(
            "Method _build_search_alg must be implemented in child class."
        )

    def _build_sheduler(self, max_t: int) -> ASHAScheduler:
        """Return the trial scheduler.

        Parameters
        -------------------
        max_t: int,
            Maximum number of steps.

        Returns
        -------------------
        Search algorithm.
        """
        return ASHAScheduler(
            time_attr='training_iteration',
            max_t=max_t,
            grace_period=10
        )

    def tune(
        self,
        train: Tuple[np.ndarray],
        validation_data: Tuple[np.ndarray],
        name: str,
        num_samples: int,
        epochs: int = 100,
        batch_size: int = 256,
        monitor: str = "AUPRC",
        mode: str = "max",
        patience: int = 5,
        min_delta: float = 0.001,
        verbose: int = 1,
        total_threads: int = None,
        fail_fast: int = True,
        keras_optimizer: str = "nadam",
        keras_loss: str = "binary_crossentropy",
        **kwargs: Dict
    ) -> pd.DataFrame:
        """Execute tuning of dataframe."""
        if total_threads is None:
            total_threads = cpu_count()
        self._analysis = tune.run(
            tune.with_parameters(
                self.fit,
                train=train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                monitor=monitor,
                mode=mode,
                patience=patience,
                min_delta=min_delta,
                optimizer=keras_optimizer,
                loss=keras_loss,
                callbacks=(TuneReportCallback(),),
                subgpu_training=True,
            ),
            metric=self._metric,
            mode=self._mode,
            name=name,
            search_alg=self._build_search_alg(**kwargs),
            scheduler=self._build_sheduler(epochs),
            resources_per_trial={
                "cpu": cpu_count()/total_threads,
                "gpu": get_minimum_gpu_rate_per_trial(total_threads)
            },
            num_samples=num_samples,
            stop=TrialPlateauStopper(
                metric=self._metric
            ),
            fail_fast=fail_fast,
            verbose=verbose
        )
        self._optimal_config = {
            key.split("/")[1]: value
            for key, value in self._analysis.dataframe().sort_values(
                self._metric,
                ascending=self._mode == "min"
            ).iloc[0].to_dict().items()
            if key.startswith("config")
        }
        return self._analysis.dataframe()
