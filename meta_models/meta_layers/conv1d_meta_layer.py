"""Class implementing meta-model for a Conv1D Layer."""
from typing import Dict

from tensorflow.keras.layers import (Activation, BatchNormalization, Conv1D,
                                     Layer)

from .regularized_meta_layer import RegularizedMetaLayer


class Conv1DMetaLayer(RegularizedMetaLayer):
    """Class implementing meta-layer for flat convolutional layers.

    Private members
    ------------------------
    _min_filters: int,
        Minimum number of filters to use for the layer.
    _max_filters: int,
        Maximum number of filters to use for the layer.
    _min_kernel_size: int,
        Minimum number of kernel size for the flat kernel.
    _max_kernel_size: int,
        Maximum number of kernel size for the flat kernel.
    _activation: str,
        The activation function to use for the layer.
    """

    def __init__(
        self,
        min_filters: int = 0,
        max_filters: int = 256,
        min_kernel_size: int = 1,
        max_kernel_size: int = 12,
        activation: str = "relu",
        **kwargs: Dict
    ):
        """Create new Conv1DResidualLayer meta-model object.

        Parameters
        ----------------------
        min_filters: int = 0,
            Minimum number of filters (neurons) in each layer.
            If the tuning process passes 0, then the layer is skipped.
        max_filters: int = 256,
            Maximum number of filters (neurons) in each layer.
        min_kernel_size: int = 1,
            Minimum size of the kernel.
        max_kernel_size: int = 12,
            Maximum size of the kernel.
        activation: str = "relu",
            The activation function to use for the layer.
        **kwargs: Dict,
            Dictionary of keyword parameters to be passed to parent class.
        """
        super().__init__(**kwargs)
        self._min_filters = min_filters
        self._max_filters = max_filters
        self._min_kernel_size = min_kernel_size
        self._max_kernel_size = max_kernel_size
        self._activation = activation

    def _space(self) -> Dict:
        """Return hyper parameters of the layer."""
        return {
            "filters": (self._min_filters, self._max_filters),
            "kernel_size": (self._min_kernel_size, self._max_kernel_size),
            **super()._space()
        }

    def _build(
        self,
        input_layers: Layer,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        **kwargs: Dict
    ) -> Layer:
        """Return built Conv1D layer block.

        If the given filters number is equal to 0, the layer is skipped.

        Parameters
        --------------------------
        input_layers: Layer,
            The input layer of the current layer.
        filters: int,
            The number of neurons of the layer.
        kernel_size: int,
            The dimension of the kernel for the layer.
        strides: int = 1,
            Strides for the convolutional layer.
        **kwargs: Dict,
            The kwargs to pass to the kernel regularizers.

        Returns
        --------------------------
        Output layer of the block.
        """
        filters = int(filters)
        kernel_size = int(kernel_size)
        if filters == 0:
            return input_layers
        layer = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            **self._build_regularizers(**kwargs)
        )(input_layers)
        if self._batch_normalization:
            layer = BatchNormalization()(layer)
        activation = Activation(self._activation)(layer)
        return activation
