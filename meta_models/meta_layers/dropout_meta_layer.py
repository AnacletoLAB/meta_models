"""Class wrapper for Keras Dropout layers usable as meta-layers.

The porpose of this layer is to provide a simple way to use Dropout layers
in meta-models, allowing for tuning the dropout parameter.
"""
from typing import Dict

from tensorflow.keras.layers import Dropout, Layer
import numpy as np

from .meta_layer import MetaLayer
from ..utils import distributions


class DropoutMetaLayer(MetaLayer):
    """Class implementing meta-layer for Dropout layers.

    Private members
    -------------------------------
    _min_dropout_rate: float,
        Minimum value for dropout rate.
    _max_dropout_rate: float,
        Maximum value for dropout rate.
    """

    def __init__(
        self,
        min_dropout_rate: float = 0,
        max_dropout_rate: float = 0.5,
        enabled: bool = True,
        **kwargs: Dict
    ):
        """Create new DropoutMetaLayer object.

        Parameters
        ------------------------
        min_dropout_rate: float = 0,
            Minimum value of dropout.
            If the tuning process passes 0, then the dropout layer is skipped.
        max_dropout_rate: float = 0.5,
            Maximum value of dropout.
            If the tuning process passes 0, then the dropout layer is skipped.
        enabled: bool = True,
            Wether to enable the dropout layer.
        **kwargs: Dict,
            Dictionary of keyword arguments to pass to parent class.
        """
        super().__init__(**kwargs)
        if not enabled:
            max_dropout_rate = min_dropout_rate = 0
        self._min_dropout_rate = min_dropout_rate
        self._max_dropout_rate = max_dropout_rate

    def _space(self) -> Dict:
        """Return space of hyper-parameters of the layer."""
        return {
            "dropout_rate": (distributions.real, self._min_dropout_rate, self._max_dropout_rate)
        }

    def _build(self, dropout_rate: float, input_layers: Layer, **kwargs) -> Layer:
        """Build Dropout layer.

        Parameters
        --------------------------
        dropout_rate: float,
            The rate of dropout.
            If the value is very close to 0, the layer is not added.
        input_layers: Layer,
            The input layer of the current layer.

        Returns
        --------------------------
        Built Dropout layer.
        """
        if np.isclose(dropout_rate, 0):
            return input_layers
        return Dropout(dropout_rate)(input_layers)
