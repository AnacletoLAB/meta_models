"""Abstract class implementing FFNN MetaModel."""
from typing import Dict, Tuple, Union

from ..meta_layers import DenseResidualMetaLayer, HeadMetaLayer, InputMetaLayer
from .meta_model import MetaModel


class ResidualFFNNMetaModel(MetaModel):
    """Class implementing FFNNMetaModel."""

    def __init__(
        self,
        input_shape: Union[int, Tuple[int]],
        blocks: int = 4,
        min_layers: int = 0,
        max_layers: int = 5,
        min_units: int = 0,
        max_units: int = 512,
        activation: str = "relu",
        output_units: int = 1,
        output_activation: int = "sigmoid",
        min_l1_regularization: float = 0,
        max_l1_regularization: float = 0.01,
        min_l2_regularization: float = 0,
        max_l2_regularization: float = 0.01,
        batch_normalization: bool = False,
        activity_regularizer: bool = False,
        kernel_regularizer: bool = False,
        bias_regularizer: bool = False,
        input_name: str = None,
        headless: bool = False
    ):
        """Create new FFNNMetaModel object.

        Parameters
        -----------------------
        input_shape: Union[int, Tuple[int]],
            The input shape of the layer.
            If an integer is provided it will be converted to a tuple.
        blocks: int = 4,
            Number of blocks of the network.
        min_layers: int = 0,
            Minimum number of layers in rectangle.
            If the tuning process passes 0, then the layer is skipped.
        max_layers: int = 5,
            Maximum number of layers in rectangle.
        min_units: int = 0,
            Minimum number of units (neurons) in each layer.
            If the tuning process passes 0, then the layer is skipped.
        max_units: int = 512,
            Maximum number of units (neurons) in each layer.
        activation: str = "relu",
            The activation function to use for the layer.
        output_units: int = 1,
            Number of neurons of the output layer.
        output_activation: int = "sigmoid",
            Activation function of the output layer.
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
        batch_normalization: bool = True,
            Wethever to use or not batch normalization.
        activity_regularizer: bool = True,
            Wethever to use an activity regularizer.
        kernel_regularizer: bool = False,
            Wethever to use a kernel regularizer.
        bias_regularizer: bool = False,
            Wethever to use a bias regularizer.
        input_name: str = None,
            Name of the input layer. This value is often used in the context
            of multimodal neural networks, otherwise is pretty meaningless
            if not for help in readability in the model summary dump.
        headless: bool = False,
            Wethever to return the model as headless.
        """
        self._blocks = blocks
        self._input_shape = input_shape
        self._input_name = input_name
        self._min_layers = min_layers
        self._max_layers = max_layers
        self._min_units = min_units
        self._max_units = max_units
        self._activation = activation
        self._output_units = output_units
        self._output_activation = output_activation
        self._min_l1_regularization = min_l1_regularization
        self._max_l1_regularization = max_l1_regularization
        self._min_l2_regularization = min_l2_regularization
        self._max_l2_regularization = max_l2_regularization
        self._batch_normalization = batch_normalization
        self._activity_regularizer = activity_regularizer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._headless = headless
        super().__init__()

    def _space(self) -> Dict:
        """Return hyper-parameters space for the model."""
        return {}

    def _structure(self):
        """Create structure of the model."""
        hidden = input_layer = InputMetaLayer(
            input_shape=self._input_shape,
            name=self._input_name
        )
        for _ in range(self._blocks):
            hidden = DenseResidualMetaLayer(
                min_layers=self._min_layers,
                max_layers=self._max_layers,
                min_units=self._min_units,
                max_units=self._max_units,
                activation=self._activation,
                min_l1_regularization=self._min_l1_regularization,
                max_l1_regularization=self._max_l1_regularization,
                min_l2_regularization=self._min_l2_regularization,
                max_l2_regularization=self._max_l2_regularization,
                batch_normalization=self._batch_normalization,
                activity_regularizer=self._activity_regularizer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer
            )(hidden)

        output_layer = hidden if self._headless else HeadMetaLayer(
            units=self._output_units,
            activation=self._output_activation
        )(hidden)

        return input_layer, output_layer
