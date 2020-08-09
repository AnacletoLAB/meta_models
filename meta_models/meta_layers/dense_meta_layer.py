"""Class implementing meta-model for a Dense Layer."""
from collections import ChainMap
from typing import Dict, List

from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
                                     Layer)

from .meta_layer import MetaLayer


class DenseMetaLayer(MetaLayer):

    def __init__(
        self,
        min_units: int = 0,
        max_units: int = 512,
        activation: str = "relu",
        min_l1_regularization: float = 0,
        max_l1_regularization: float = 0.01,
        min_l2_regularization: float = 0,
        max_l2_regularization: float = 0.01,
        batch_normalization: bool = False,
        activity_regularizer: bool = False,
        kernel_regularizer: bool = False,
        bias_regularizer: bool = False,
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
        min_l1_regularization: float = 0,
            Minimum value of l1 regularization.
            If the tuning process passes 0, then the regularization is skipped.
            This is the minimum value used for all form of regularization,
            but diffent keyword arguments are used depending on what is
            enabled in this layer, so different values will be passed by the
            optimization process. Rarily the regularization layers have vastly
            different values, hence the absence of multiple parameters.
        max_l1_regularization: float = 0.01,
            Maximum value of l1 regularization.
            This is the maximum value used for all form of regularization,
            but diffent keyword arguments are used depending on what is
            enabled in this layer, so different values will be passed by the
            optimization process. Rarily the regularization layers have vastly
            different values, hence the absence of multiple parameters.
        min_l2_regularization: float = 0,
            Minimum value of l2 regularization.
            If the tuning process passes 0, then the regularization is skipped.
            This is the minimum value used for all form of regularization,
            but diffent keyword arguments are used depending on what is
            enabled in this layer, so different values will be passed by the
            optimization process. Rarily the regularization layers have vastly
            different values, hence the absence of multiple parameters.
        max_l2_regularization: float = 0.01,
            Maximum value of l2 regularization.
            This is the maximum value used for all form of regularization,
            but diffent keyword arguments are used depending on what is
            enabled in this layer, so different values will be passed by the
            optimization process. Rarily the regularization layers have vastly
            different values, hence the absence of multiple parameters.
        batch_normalization: bool = False,
            Wethever to use or not batch normalization.
        activity_regularizer: bool = False,
            Wethever to use an activity regularizer.
        kernel_regularizer: bool = False,
            Wethever to use a kernel regularizer.
        bias_regularizer: bool = False,
            Wethever to use a bias regularizer.
        **kwargs: Dict,
            Dictionary of keyword parameters to be passed to parent class.
        """
        super().__init__(**kwargs)
        self._min_units = min_units
        self._max_units = max_units
        self._activation = activation
        self._min_l1_regularization = min_l1_regularization
        self._max_l1_regularization = max_l1_regularization
        self._min_l2_regularization = min_l2_regularization
        self._max_l2_regularization = max_l2_regularization
        self._batch_normalization = batch_normalization
        self._activity_regularizer = activity_regularizer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer

    def _space(self) -> Dict:
        """Return hyper parameters of the layer."""
        return {
            "units": (self._min_units, self._max_units),
            **ChainMap(*[
                {
                    "{regularizer}_{reg_type}".format(): reg_range
                    for reg_type, reg_range in zip(
                        ("l1", "l1"),
                        (
                            (
                                self._min_l1_regularization,
                                self._max_l1_regularization
                            ),
                            (
                                self._min_l2_regularization,
                                self._max_l2_regularization
                            )
                        )
                    )
                }
                for regularizer, enabled in zip(
                    (
                        "activity_regularizer",
                        "kernel_regularizer",
                        "bias_regularizer"
                    ),
                    (
                        self._activity_regularizer,
                        self._kernel_regularizer,
                        self._bias_regularizer
                    )
                )
                if enabled
            ])
        }

    def _build(self, input_layers: Layer, units: int) -> Layer:
        """Return built Dense layer block.

        If the given units number is equal to 0, the layer is skipped.

        Parameters
        --------------------------
        input_layers: Layer,
            The input layer of the current layer.
        units: int,
            The number of neurons of the layer.

        Returns
        --------------------------
        Output layer of the block.
        """
        units = int(units)
        if units == 0:
            return input_layers
        layer = Dense(units=units)(input_layers)
        if self._batch_normalization:
            layer = BatchNormalization()(layer)
        activation = Activation(self._activation)(layer)
        return activation
