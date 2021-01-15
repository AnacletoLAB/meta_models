"""Class implementing abstract RayBayesianOptimizationTuner."""
from typing import Dict

from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch

from ..meta_models import MetaModel
from ..utils import distributions
from .ray_tuner import RayTuner


class RayBayesianOptimizationTuner(RayTuner):

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

    def _build_search_alg(
        self,
        space: Dict,
        random_search_steps: int
    ) -> BayesOptSearch:
        """Create Bayesian Algorithm search method.

        Parameters
        -------------------
        space: Dict,
            Space of hyper-parameters.
        random_search_steps: int,
            Number of random search steps.

        Returns
        -------------------
        Instance of BayesOptSearch. 
        """
        space = {
            key: (values[1], values[2])
            for key, values in space.items()
            if values[0] in (distributions.real, distributions.integer)
        }
        return BayesOptSearch(
            space,
            metric=self._metric,
            mode=self._mode,
            random_search_steps=random_search_steps,
            random_state=self._random_state,
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
            grace_period=10,
            reduction_factor=3
        )