"""Class wrapper for Keras MaxPool3D layers usable as meta-layers."""
from typing import Dict

from tensorflow.keras.layers import MaxPool3D, Layer

from .meta_layer import MetaLayer
from ..utils import distributions


class MaxPool3DMetaLayer(MetaLayer):
    """Class implementing meta-layer for MaxPool3D layers.

    Private members
    -------------------------------
    _min_x_pool_size: int,
        Minimum value for pool rate.
    _max_x_pool_size: int,
        Maximum value for pool rate.
    _min_y_pool_size: int,
        Minimum value for pool rate.
    _max_y_pool_size: int,
        Maximum value for pool rate.
    _min_z_pool_size: int,
        Minimum value for pool rate.
    _max_z_pool_size: int,
        Maximum value for pool rate.
    """

    def __init__(
        self,
        min_x_pool_size: int = 1,
        max_x_pool_size: int = 8,
        min_y_pool_size: int = 1,
        max_y_pool_size: int = 4,
        min_z_pool_size: int = 1,
        max_z_pool_size: int = 4,
        enabled: bool = True,
        **kwargs: Dict
    ):
        """Create new MaxPool3DMetaLayer object.

        Parameters
        ------------------------
        min_x_pool_size: int = 1,
            Minimum value of pool size.
            If the tuning process passes 0, then the maxpooling layer is skipped.
        max_x_pool_size: int = 8,
            Maximum value of pool size.
            If the tuning process passes 0, then the maxpooling layer is skipped.
        min_y_pool_size: int = 1,
            Minimum value of pool size.
            If the tuning process passes 0, then the maxpooling layer is skipped.
        max_y_pool_size: int = 4,
            Maximum value of pool size.
            If the tuning process passes 0, then the maxpooling layer is skipped.
        min_z_pool_size: int = 1,
            Minimum value of pool size.
            If the tuning process passes 0, then the maxpooling layer is skipped.
        max_z_pool_size: int = 4,
            Maximum value of pool size.
            If the tuning process passes 0, then the maxpooling layer is skipped.
        enabled: bool = True,
            Wether to enable the maxpooling 2d layer.
        **kwargs: Dict,
            Dictionary of keyword arguments to pass to parent class.
        """
        super().__init__(**kwargs)
        if not enabled:
            max_x_pool_size = min_x_pool_size = 0
            max_y_pool_size = min_y_pool_size = 0
            max_z_pool_size = min_z_pool_size = 0
        self._min_x_pool_size = min_x_pool_size
        self._max_x_pool_size = max_x_pool_size
        self._min_y_pool_size = min_y_pool_size
        self._max_y_pool_size = max_y_pool_size
        self._min_z_pool_size = min_z_pool_size
        self._max_z_pool_size = max_z_pool_size

    def _space(self) -> Dict:
        """Return space of hyper-parameters of the layer."""
        return {
            "x_pool_size": (distributions.integer, self._min_x_pool_size, self._max_x_pool_size),
            "y_pool_size": (distributions.integer, self._min_y_pool_size, self._max_y_pool_size),
            "z_pool_size": (distributions.integer, self._min_z_pool_size, self._max_z_pool_size),
        }

    def _build(
        self,
        x_pool_size: int,
        y_pool_size: int,
        z_pool_size: int,
        input_layers: Layer,
        **kwargs
    ) -> Layer:
        """Build MaxPool3D layer.

        Parameters
        --------------------------
        x_pool_size: float,
            The rate of pool size for the x axis.
            If the value is 0, the layer is not added.
        y_pool_size: float,
            The rate of pool size for the y axis.
            If the value is 0, the layer is not added.
        z_pool_size: float,
            The rate of pool size for the z axis.
            If the value is 0, the layer is not added.
        input_layers: Layer,
            The input layer of the current layer.

        Returns
        --------------------------
        Built MaxPool3D layer.
        """
        x_pool_size = round(x_pool_size)
        y_pool_size = round(y_pool_size)
        z_pool_size = round(z_pool_size)
        if any(
            pool_size == 0
            for pool_size in (x_pool_size, y_pool_size, z_pool_size)
        ):
            return input_layers
        return MaxPool3D(pool_size=(
            x_pool_size,
            y_pool_size,
            z_pool_size
        ), padding="same",)(input_layers)
