"""Class implementing abstract RayBayesianOptimizationTuner."""
from typing import Dict

from ray.tune.schedulers.pb2 import PB2
import numpy as np
from ..meta_models import MetaModel
from ..utils import distributions
from .ray_tuner import RayTuner


class RayPopulationBasedBanditsTuner(RayTuner):

    def __init__(
        self,
        meta_model: MetaModel,
        metric: str = "val_AUPRC",
        mode: str = "max",
        random_state: int = 42,
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
        random_state: int = 42,
            Random state to reproduce the tuning procedure.
        """
        super().__init__(
            meta_model=meta_model,
            metric=metric,
            mode=mode
        )
        self._random_state = random_state

    def _parse_space(self) -> Dict:
        """Return the training space adapted for the considered algorithm.

        Returns
        -------------------
        Search algorithm.
        """
        return {
            key: [values[1], values[2]]
            for key, values in self._meta_model.space().items()
            if values[0] in (distributions.real, distributions.integer)
        }

    def _build_search_alg(
        self,
        **kwargs
    ):
        """Return None as no search algorithm is used."""
        return None

    def _build_sheduler(self, *args, **kwargs) -> PB2:
        """Return the trial scheduler.

        Returns
        -------------------
        Search algorithm.
        """
        return PB2(
            time_attr='time_total_s',
            perturbation_interval=1000.0,
            hyperparam_bounds=self._parse_space()
        )
