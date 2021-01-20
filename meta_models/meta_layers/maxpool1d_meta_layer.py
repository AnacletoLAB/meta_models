"""Class wrapper for Keras MaxPool1D layers usable as meta-layers."""
from typing import Dict

from tensorflow.keras.layers import MaxPool1D, Layer

from .meta_layer import MetaLayer
from ..utils import distributions


class MaxPool1DMetaLayer(MetaLayer):
    """Class implementing meta-layer for MaxPool1D layers.

    Private members
    -------------------------------
    _min_pool_size: int,
        Minimum value for pool rate.
    _max_pool_size: int,
        Maximum value for pool rate.
    """

    def __init__(
        self,
        min_pool_size: int = 1,
        max_pool_size: int = 8,
        enabled: bool = True,
        **kwargs: Dict
    ):
        """Create new MaxPool1DMetaLayer object.

        Parameters
        ------------------------
        min_pool_size: int = 1,
            Minimum value of pool size.
            If the tuning process passes 0, then the maxpooling layer is skipped.
        max_pool_size: int = 8,
            Maximum value of pool size.
            If the tuning process passes 0, then the maxpooling layer is skipped.
        enabled: bool = True,
            Wether to enable the maxpooling layer.
        **kwargs: Dict,
            Dictionary of keyword arguments to pass to parent class.
        """
        super().__init__(**kwargs)
        if not enabled:
            max_pool_size = min_pool_size = 0
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size

    def _space(self) -> Dict:
        """Return space of hyper-parameters of the layer."""
        return {
            "pool_size": (distributions.integer, self._min_pool_size, self._max_pool_size)
        }

    def _build(self, pool_size: int, input_layers: Layer, **kwargs) -> Layer:
        """Build MaxPool1D layer.

        Parameters
        --------------------------
        pool_size: float,
            The rate of pool size.
            If the value is 0, the layer is not added.
        input_layers: Layer,
            The input layer of the current layer.

        Returns
        --------------------------
        Built MaxPool1D layer.
        """
        pool_size = round(pool_size)
        if pool_size == 0:
            return input_layers
        return MaxPool1D(pool_size, padding="same")(input_layers)
