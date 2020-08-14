"""Class wrapper for Keras Concatenate layers usable as meta-layers."""
from .meta_layer import MetaLayer
from typing import Dict, Union, Tuple, List
from tensorflow.keras.layers import Input, Layer, Concatenate


class ConcatenateMetaLayer(MetaLayer):
    """Class implementing Concatenation Meta Layer.
    
    The pourpose of a concatenation meta layer is to concatenate the output
    of the multiple meta-layers (tipically from different meta-models).
    """

    def __init__(self, **kwargs):
        """Create new ConcatenateMetaLayer object.

        Parameters
        ------------------------
        **kwargs: Dict,
            Dictionary of keyword arguments to pass to parent class.
        """
        super().__init__(**kwargs)

    def _space(self) -> Dict:
        """Return space of hyper-parameters of the layer."""
        return {}

    def _build(self, input_layers: List[Layer], **kwargs) -> Layer:
        """Build input layer."""
        return Concatenate()([input_layers])
