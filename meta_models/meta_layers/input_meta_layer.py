"""Class wrapper for Keras Input layers usable as meta-layers.

The porpose of this layer is to provide a simple way to use input layers
in meta-models, but it does not provide any additional particular parameter.
"""
from .meta_layer import MetaLayer
from typing import Dict, Union, Tuple
from tensorflow.keras.layers import Input, Layer


class InputMetaLayer(MetaLayer):

    def __init__(self, input_shape: Union[int, Tuple[int]], name: str = None, **kwargs):
        """Create new InputMetaLayer object.

        This is currently not a complete wrapper on the Layer Input, as it
        only wraps the keyword arguments that are used in most common models.

        Parameters
        ------------------------
        input_shape: Union[int, Tuple[int]],
            The input shape of the layer.
            If an integer is provided it will be converted to a tuple.
        name: str = None,
            Name of the input layer. This value is often used in the context
            of multimodal neural networks, otherwise is pretty meaningless
            if not for help in readability in the model summary dump.
        **kwargs: Dict,
            Dictionary of keyword arguments to pass to parent class.
        """
        super().__init__(**kwargs)
        if isinstance(input_shape, int):
            input_shape = (input_shape, )
        self._input_shape = input_shape
        self._name = name

    def _space(self) -> Dict:
        """Return space of hyper-parameters of the layer."""
        return {}

    def _build(self, **kwargs) -> Layer:
        """Build input layer."""
        return Input(
            shape=self._input_shape,
            name=self._name
        )
