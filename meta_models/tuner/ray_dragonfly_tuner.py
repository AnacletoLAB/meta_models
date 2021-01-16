"""Class implementing abstract RayBayesianOptimizationTuner."""
from typing import Dict

from ray.tune.suggest.dragonfly import DragonflySearch

from ..meta_models import MetaModel
from ..utils import distributions
from .ray_tuner import RayTuner


class RayDragonflyTuner(RayTuner):

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
        return [
            {
                "name": key,
                "type": "float",
                "min":values[1],
                "max": values[2],
            }
            if distributions.real == values[0]
            else {
                "name": key,
                "type": "int",
                "min":values[1],
                "max": values[2],
            }
            for key, values in self._meta_model.space().items()
            if values[0] in (distributions.real, distributions.integer)
        ]

    def _build_search_alg(
        self,
        optimizer: str = "bandit",
        domain: str = "euclidean",
    ) -> DragonflySearch:
        """Create Ax search method.

        Parameters
        -------------------
        optimizer: str = "bandit",
            Optimizer provided from dragonfly.
            Choose an optimiser that extends BlackboxOptimiser.
            If this is a string, domain must be set and optimizer must be one of [random, bandit, genetic].
        domain: str = "euclidean",
            Optional domain.
            Should only be set if you don’t pass an optimizer as the optimizer argument.
            Has to be one of [cartesian, euclidean].

        Returns
        -------------------
        Instance of DragonflySearch. 
        """
        return DragonflySearch(
            optimizer=optimizer,
            domain=domain,
            space=self._parse_space(),
            metric=self._metric,
            mode=self._mode
        )
