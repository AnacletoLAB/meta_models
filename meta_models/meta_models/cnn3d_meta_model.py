"""Abstract class implementing FFNN MetaModel."""
from typing import Dict, Tuple, Union

from ..meta_layers import (Conv3DRectangularMetaLayer,
                           HeadMetaLayer, InputMetaLayer, FlattenMetaLayer)
from .meta_model import MetaModel
from .ffnn_meta_model import FFNNMetaModel


class CNN3DMetaModel(MetaModel):
    """Class implementing CNN3DMetaModel.

    The class implements a meta-model for 3D CNN, useful for handling some
    kinds of multi-channel image data. The meta-model is tipically composed of a few
    convolutional blocks, with optional residuality, with a sequence of Dense
    blocks on top and finally a head layer with, tipically, a single neuron
    and sigmoid activation when using the model on a binary classification task.

    The class is also meant to be used as a sub-module in the context of a
    multi-modal neural network.

    Private members
    --------------------------
    _input_shape: Union[int, Tuple[int]],
        The input shape of the layer.
        If an integer is provided it will be converted to a tuple.
    _blocks: int,
        Number of blocks of the network.
    _conv2d_meta_layer_kwargs: Dict,
        Keyword arguments to pass to the builder of  Conv2D Meta Layers.
    _top_ffnn_meta_model_kwargs: Dict,
        Keyword arguments to pass to the builder of  FFNN Meta Models.
    _input_name: str,
        Name of the input layer. This value is often used in the context
        of multimodal neural networks, otherwise is pretty meaningless
        if not for help in readability in the model summary dump.
    """

    def __init__(
        self,
        input_shape: Union[int, Tuple[int]],
        blocks: int = 4,
        meta_layer_kwargs: Dict = None,
        top_ffnn_meta_model_kwargs: Dict = None,
        input_name: str = None
    ):
        """Create new CNN3DMetaModel object.

        Parameters
        -----------------------
        input_shape: Union[int, Tuple[int]],
            The input shape of the layer.
            If an integer is provided it will be converted to a tuple.
        blocks: int = 4,
            Number of blocks of the network.
        conv3d_meta_layer_kwargs: Dict = None,
            Keyword arguments to pass to the builder of  Conv3D Meta Layers.
        top_ffnn_meta_model_kwargs: Dict = None,
            Keyword arguments to pass to the builder of  FFNN Meta Models.
        input_name: str = None,
            Name of the input layer. This value is often used in the context
            of multimodal neural networks, otherwise is pretty meaningless
            if not for help in readability in the model summary dump.
        """
        self._blocks = blocks
        self._input_shape = input_shape
        self._input_name = input_name
        self._meta_layer_kwargs = {} if meta_layer_kwargs is None else meta_layer_kwargs
        self._top_ffnn = FFNNMetaModel(**(
            {}
            if top_ffnn_meta_model_kwargs is None
            else top_ffnn_meta_model_kwargs
        ))
        super().__init__()

    def _space(self) -> Dict:
        """Return hyper-parameters space for the model."""
        return {
            **self._top_ffnn._space()
        }

    def _structure(self):
        """Create structure of the model."""
        hidden = input_layer = InputMetaLayer(
            input_shape=self._input_shape,
            name=self._input_name
        )
        for _ in range(self._blocks):
            hidden = Conv3DRectangularMetaLayer(
                **self._meta_layer_kwargs
            )(hidden)

        hidden = FlattenMetaLayer()(hidden)
        _, output_layer = self._top_ffnn._structure(hidden)

        return input_layer, output_layer
