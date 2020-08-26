"""Abstract class implementing FFNN MetaModel."""
from typing import Dict, Tuple, Union

from ..meta_layers import (DenseRectangularMetaLayer, HeadMetaLayer,
                           InputMetaLayer)
from .meta_model import MetaModel


class FFNNMetaModel(MetaModel):
    """Class implementing FFNNMetaModel.

    The class implements a meta-model for FFNN, useful for handling some
    kinds of vectorial data. The meta-model is tipically composed of a few
    dense blocks, with optional residuality and a head layer on top with,
    tipically, a single neuron and sigmoid activation when using the model
    on a binary classification task.

    The class is also meant to be used as a sub-module in the context of a
    multi-modal neural network.

    Private members
    -----------------------------
    _input_shape: Union[int, Tuple[int]] = None,
        The input shape of the layer.
        If an integer is provided it will be converted to a tuple.
        If no input_shape is provided then an InputMetaLayer must
        be passed in the structure method.
    _blocks: int = 4,
        Number of blocks of the network.
    _activation: str = "relu",
        The activation function to use for the layer.
    _output_units: int = 1,
        Number of neurons of the output layer.
    _meta_layer_kwargs: Dict = None,
        Keyword arguments to pass to the builder of Dense Meta Layers.
    _input_name: str = None,
        Name of the input layer. This value is often used in the context
        of multimodal neural networks, otherwise is pretty meaningless
        if not for help in readability in the model summary dump.
    _headless: bool = False,
        Wethever to return the model as headless.
    """

    def __init__(
        self,
        input_shape: Union[int, Tuple[int]] = None,
        blocks: int = 4,
        output_units: int = 1,
        output_activation: int = "sigmoid",
        meta_layer_kwargs: Dict = None,
        input_name: str = None,
        headless: bool = False
    ):
        """Create new FFNNMetaModel object.

        Parameters
        -----------------------
        input_shape: Union[int, Tuple[int]] = None,
            The input shape of the layer.
            If an integer is provided it will be converted to a tuple.
            If no input_shape is provided then an InputMetaLayer must
            be passed in the structure method.
        blocks: int = 4,
            Number of blocks of the network.
        activation: str = "relu",
            The activation function to use for the layer.
        output_units: int = 1,
            Number of neurons of the output layer.
        meta_layer_kwargs: Dict = None,
            Keyword arguments to pass to the builder of Dense Meta Layers.
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
        self._output_units = output_units
        self._output_activation = output_activation
        self._meta_layer_kwargs = {} if meta_layer_kwargs is None else meta_layer_kwargs
        self._headless = headless
        super().__init__()

    def _space(self) -> Dict:
        """Return hyper-parameters space for the model."""
        return {}

    def structure(self, input_layer: InputMetaLayer = None):
        """Create structure of the model."""
        hidden = input_layer = InputMetaLayer(
            input_shape=self._input_shape,
            name=self._input_name
        ) if input_layer is None else input_layer

        for _ in range(self._blocks):
            hidden = DenseRectangularMetaLayer(
                **self._meta_layer_kwargs
            )(hidden)

        output_layer = hidden if self._headless else HeadMetaLayer(
            units=self._output_units,
            activation=self._output_activation
        )(hidden)

        return input_layer, output_layer
