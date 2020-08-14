"""Abstract class implementing FFNN MetaModel."""
from typing import Dict, Tuple, Union

from ..meta_layers import Conv2DRectangularMetaLayer, HeadMetaLayer, InputMetaLayer
from .meta_model import MetaModel
from .ffnn_meta_model import FFNNMetaModel


class CNN2DMetaModel(MetaModel):
    """Class implementing CNN2DMetaModel.
    
    !TODO: Add docstrings for class.

    """

    def __init__(
        self,
        input_shape: Union[int, Tuple[int]],
        blocks: int = 4,
        meta_layer_kwargs: Dict = None,
        top_ffnn_meta_model_kwargs: Dict = None,
        input_name: str = None
    ):
        """Create new CNN2DMetaModel object.

        Parameters
        -----------------------
        input_shape: Union[int, Tuple[int]],
            The input shape of the layer.
            If an integer is provided it will be converted to a tuple.
        blocks: int = 4,
            Number of blocks of the network.
        conv2d_meta_layer_kwargs: Dict = None,
            Keyword arguments to pass to the builder of  Conv2D Meta Layers.
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
            hidden = Conv2DRectangularMetaLayer(
                **self._meta_layer_kwargs
            )(hidden)

        _, output_layer = self._top_ffnn._structure(hidden)

        return input_layer, output_layer
