"""Class implementing abstract RayHyperOptTuner."""
from typing import Dict

from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

from ..meta_models import MetaModel
from ..utils import distributions
from .ray_tuner import RayTuner


class RayHyperOptTuner(RayTuner):

    def __init__(
        self,
        meta_model: MetaModel,
        metric: str = "val_loss",
        mode: str = "min",
        random_state: int = 42,
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
            key: hp.uniform(key, values[1], values[2])
            if distributions.real == values[0]
            else hp.randint(key, values[1], values[2])
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
