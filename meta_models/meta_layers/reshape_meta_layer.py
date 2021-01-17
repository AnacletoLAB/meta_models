"""Class wrapper for Keras Reshape layers usable as meta-layers."""
from typing import Dict, List, Union, Tuple

from tensorflow.keras.layers import Reshape, Layer

from .meta_layer import MetaLayer


class ReshapeMetaLayer(MetaLayer):
    """Class implementing Concatenation Meta Layer.

    The pourpose of a concatenation meta layer is to Reshape the output
    of the multiple meta-layers (tipically from different meta-models).
    """

    def __init__(self, target_shape: Union[int, Tuple[int]], **kwargs):
        """Create new ReshapeMetaLayer object.

        Parameters
        ------------------------
        target_shape: Union[int, Tuple[int]] = None,
            Shape to modify input into.
        **kwargs: Dict,
            Dictionary of keyword arguments to pass to parent class.
        """
        super().__init__(**kwargs)
        self._target_shape = target_shape

    def _space(self) -> Dict:
        """Return space of hyper-parameters of the layer."""
        return {}

    def _build(self, input_layers: List[Layer], **kwargs) -> Layer:
        """Build input layer."""
        return Reshape(
            target_shape=self._target_shape
        )(input_layers)
