"""Class implementing abstract Tuner."""
from typing import Dict, Callable
import numpy as np
from tensorflow.keras.models import Model
from ..meta_models import MetaModel


class Tuner:
    """Class implementing an abstract Tuner object.

    An abstract Tuner object tunes a meta-model and determines
    the near-optimal parameters of the model for the considered task.
    """

    def __init__(
        self,
        meta_model: MetaModel,
        score: Callable[[Model, int], float],
        holdouts: int = 1,
    ):
        """Create the Tuner object.

        Parameters
        --------------------------
        meta_model: MetaModel,
            The meta-model to optimize.
        score: Callable[[Model, int], float],
            The score function to call to evaluate a built model on one of
            the considered holdouts.
        holdouts: int,
            Number of internal holdouts to use per loop.

        Raises
        --------------------------
        ValueError,
            If the given holdouts number is not a strictly positive integer.
        """
        if not isinstance(holdouts, int) or holdouts < 1:
            raise ValueError(
                (
                    "Given holdouts number ({})"
                    " is not a strictly positive integer"
                ).format(holdouts)
            )
        self._meta_model = meta_model
        self._score = score
        self._holdouts = holdouts

    def tune(self):
        """Tune the model."""
        raise NotImplementedError(
            "Method tune must be implemented in child class."
        )
