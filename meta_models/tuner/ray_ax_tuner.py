"""Class implementing abstract RayBayesianOptimizationTuner."""
from typing import Dict

from ray.tune.suggest.ax import AxSearch

from ..meta_models import MetaModel
from ..utils import distributions
from .ray_tuner import RayTuner


class RayAxTuner(RayTuner):

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
        return [
            {
                "name": key,
                "type": "range",
                "bounds": [values[1], values[2]],
                "value_type":"float",
                "log_scale": False,
            }
            if distributions.real == values[0]
            else {
                "name": key,
                "type": "range",
                "bounds": [values[1], values[2]],
                "value_type":"int",
                "log_scale": False,
            }
            for key, values in self._meta_model.space().items()
            if values[0] in (distributions.real, distributions.integer)
        ]

    def _build_search_alg(
        self,
        **kwargs: Dict
    ) -> AxSearch:
        """Create Ax search method.

        Parameters
        -------------------
        kwargs: Dict,
            Additional parameters that can be specified in algorithm.

        Returns
        -------------------
        Instance of AxSearch. 
        """
        return AxSearch(
            self._parse_space(),
            metric=self._metric,
            mode=self._mode,
            **kwargs
        )
