"""Class implementing meta-model for a Conv3D Layer."""
from typing import Dict

from tensorflow.keras.layers import (Activation, BatchNormalization, Conv3D,
                                     Layer)

from .regularized_meta_layer import RegularizedMetaLayer
from ..utils import distributions


class Conv3DMetaLayer(RegularizedMetaLayer):
    """Class implementing meta-layer for tri-dimensional convolutional layers.

    Private members
    ------------------------
    _min_filters: int,
        Minimum number of filters to use for the layer.
    _max_filters: int,
        Maximum number of filters to use for the layer.
    _min_x_kernel_size: int,
        Minimum size of the kernel on the lenght axis.
    _max_x_kernel_size: int,
        Maximum size of the kernel on the lenght axis.
    _min_y_kernel_size: int,
        Minimum size of the kernel on the depth axis.
    _max_y_kernel_size: int,
        Maximum size of the kernel on the depth axis.
    _min_z_kernel_size: int,
        Minimum size of the kernel on the height axis.
    _max_z_kernel_size: int,
        Maximum size of the kernel on the height axis.
    _activation: str,
        The activation function to use for the layer.
    """

    def __init__(
        self,
        min_filters: int = 0,
        max_filters: int = 256,
        min_x_kernel_size: int = 1,
        max_x_kernel_size: int = 5,
        min_y_kernel_size: int = 1,
        max_y_kernel_size: int = 5,
        min_z_kernel_size: int = 1,
        max_z_kernel_size: int = 5,
        activation: str = "relu",
        **kwargs: Dict
    ):
        """Create new Conv3DResidualLayer meta-model object.

        Parameters
        ----------------------
        min_filters: int = 0,
            Minimum number of filters (neurons) in each layer.
            If the tuning process passes 0, then the layer is skipped.
        max_filters: int = 256,
            Maximum number of filters (neurons) in each layer.
        min_x_kernel_size: int = 1,
            Minimum size of the kernel on the lenght axis.
        max_x_kernel_size: int = 5,
            Maximum size of the kernel on the lenght axis.
        min_y_kernel_size: int = 1,
            Minimum size of the kernel on the depth axis.
        max_y_kernel_size: int = 5,
            Maximum size of the kernel on the depth axis.
        min_z_kernel_size: int = 1,
            Minimum size of the kernel on the height axis.
        max_z_kernel_size: int = 5,
            Maximum size of the kernel on the height axis.
        activation: str = "relu",
            The activation function to use for the layer.
        **kwargs: Dict,
            Dictionary of keyword parameters to be passed to parent class.
        """
        super().__init__(**kwargs)
        self._min_filters = min_filters
        self._max_filters = max_filters
        self._min_x_kernel_size = min_x_kernel_size
        self._max_x_kernel_size = max_x_kernel_size
        self._min_y_kernel_size = min_y_kernel_size
        self._max_y_kernel_size = max_y_kernel_size
        self._min_z_kernel_size = min_z_kernel_size
        self._max_z_kernel_size = max_z_kernel_size
        self._activation = activation

    def _space(self) -> Dict:
        """Return hyper parameters of the layer."""
        return {
            "filters": (distributions.integer, self._min_filters, self._max_filters),
            "x_kernel_size": (distributions.integer, self._min_x_kernel_size, self._max_x_kernel_size),
            "y_kernel_size": (distributions.integer, self._min_y_kernel_size, self._max_y_kernel_size),
            "z_kernel_size": (distributions.integer, self._min_z_kernel_size, self._max_z_kernel_size),
            **super()._space()
        }

    def _build(
        self,
        input_layers: Layer,
        filters: int,
        x_kernel_size: int,
        y_kernel_size: int,
        z_kernel_size: int,
        strides: int = (1, 1, 1),
        **kwargs: Dict
    ) -> Layer:
        """Return built Conv3D layer block.

        If the given filters number is equal to 0, the layer is skipped.

        Parameters
        --------------------------
        input_layers: Layer,
            The input layer of the current layer.
        filters: int,
            The number of neurons of the layer.
        x_kernel_size: int,
            The dimension of the kernel for the layer, on the length axis.
        y_kernel_size: int,
            The dimension of the kernel for the layer, on the depth axis.
        z_kernel_size: int,
            The dimension of the kernel for the layer, on the height axis.
        strides: int = (1, 1),
            Strides for the convolutional layer.
        **kwargs: Dict,
            The kwargs to pass to the kernel regularizers.

        Returns
        --------------------------
        Output layer of the block.
        """
        filters = round(filters)
        x_kernel_size = round(x_kernel_size)
        y_kernel_size = round(y_kernel_size)
        z_kernel_size = round(z_kernel_size)
        if filters == 0:
            return input_layers

        layer = Conv3D(
            filters=filters,
            kernel_size=(x_kernel_size, y_kernel_size, z_kernel_size),
            strides=strides,
            padding="same",
            **self._build_regularizers(**kwargs)
        )(input_layers)
        if self._batch_normalization:
            layer = BatchNormalization()(layer)
        activation = Activation(self._activation)(layer)
        return activation
