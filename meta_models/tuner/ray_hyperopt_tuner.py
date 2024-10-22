"""Class implementing abstract RayHyperOptTuner."""
from typing import Dict

from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp
import numpy as np

from ..meta_models import MetaModel
from ..utils import distributions
from .ray_tuner import RayTuner


class RayHyperOptTuner(RayTuner):

    def __init__(
        self,
        meta_model: MetaModel,
        metric: str = "val_AUPRC",
        mode: str = "max",
        random_state: int = 42,
        resolution: int = 10
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
        resolution: int = 10,
            What resolution to use for the integer values.
        """
        super().__init__(
            meta_model=meta_model,
            metric=metric,
            mode=mode
        )
        self._random_state = random_state
        self._resolution = resolution

    def _parse_space(self) -> Dict:
        """Return the training space adapted for the considered algorithm.

        Returns
        -------------------
        Search algorithm.
        """
        return {
            key: hp.uniform(key, values[1], values[2])
            if distributions.real == values[0]
            else hp.choice(key, np.unique(np.linspace(values[1], values[2], num=self._resolution).astype(int)))
            if distributions.integer == values[0]
            else hp.choice(key, values[1:])
            for key, values in self._meta_model.space().items()
        }

    def _build_search_alg(
        self,
        random_search_steps: int
    ) -> HyperOptSearch:
        """Create Bayesian Algorithm search method.

        Parameters
        -------------------
        random_search_steps: int,
            Number of random search steps.

        Returns
        -------------------
        Instance of HyperOptSearch. 
        """
        return HyperOptSearch(
            self._parse_space(),
            metric=self._metric,
            mode=self._mode,
            n_initial_points=random_search_steps,
            random_state_seed=self._random_state,
        )
