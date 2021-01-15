"""Class implementing abstract RayBayesianOptimizationTuner."""
from typing import Dict
from ray.tune.suggest.bayesopt import BayesOptSearch
from .ray_tuner import RayTuner


class RayBayesianOptimizationTuner(RayTuner):

    def __init__(
        self,
        random_search_steps: int,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._random_search_steps = random_search_steps
        self._random_state = random_state

    def _build_search_alg(self, space: Dict) -> BayesOptSearch:
        return BayesOptSearch(
            space,
            metric=self._metric,
            mode=self._mode,
            random_search_steps=self._random_search_steps,
            random_state=self._random_state,
        )
