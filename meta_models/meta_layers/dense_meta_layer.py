"""Class implementing meta-model for a Dense Layer."""
from collections import ChainMap
from typing import Dict, List

from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
                                     Layer)

from .regularized_meta_layer import RegularizedMetaLayer


class DenseMetaLayer(RegularizedMetaLayer):
    """Class implementing meta-layer for Dense layers.

    Private members
    ------------------------
    _min_units: int,
        Minimum number of units (neurons) in each layer.
        If the tuning process passes 0, then the layer is skipped.
    _max_units: int,
        Maximum number of units (neurons) in each layer.
    _activation: str,
        The activation function to use for the layer.
    """

    def __init__(
        self,
        min_units: int = 0,
        max_units: int = 512,
        activation: str = "relu",
        **kwargs: Dict
    ):
        """Create new DenseResidualLayer meta-model object.

        Parameters
        ----------------------
        min_units: int = 0,
            Minimum number of units (neurons) in each layer.
            If the tuning process passes 0, then the layer is skipped.
        max_units: int = 512,
            Maximum number of units (neurons) in each layer.
        activation: str = "relu",
            The activation function to use for the layer.
        **kwargs: Dict,
            Dictionary of keyword parameters to be passed to parent class.
        """
        super().__init__(**kwargs)
        self._min_units = min_units
        self._max_units = max_units
        self._activation = activation

    def _space(self) -> Dict:
        """Return hyper parameters of the layer."""
        return {
            "units": (self._min_units, self._max_units),
            **super()._space()
        }

    def _build(
        self,
        input_layers: Layer,
        units: int,
        **kwargs: Dict
    ) -> Layer:
        """Return built Dense layer block.

        If the given units number is equal to 0, the layer is skipped.

        Parameters
        --------------------------
        input_layers: Layer,
            The input layer of the current layer.
        units: int,
            The number of neurons of the layer.
        **kwargs: Dict,
            The kwargs to pass to the kernel regularizers.

        Returns
        --------------------------
        Output layer of the block.
        """
        units = round(units)
        if units == 0:
            return input_layers
        layer = Dense(
            units=units,
            **self._build_regularizers(**kwargs)
        )(input_layers)
        if self._batch_normalization:
            layer = BatchNormalization()(layer)
        activation = Activation(self._activation)(layer)
        return activation
