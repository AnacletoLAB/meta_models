"""Class wrapper for Keras Flatten layers usable as meta-layers.

The porpose of this layer is to provide a simple way to use flatten layers
in meta-models, but it does not provide any additional particular parameter.
"""
from .meta_layer import MetaLayer
from typing import Dict, Union, Tuple
from tensorflow.keras.layers import Flatten, Layer


class FlattenMetaLayer(MetaLayer):

    def __init__(self, **kwargs):
        """Create new FlattenMetaLayer object.

        This is currently not a complete wrapper on the Layer Flatten, as it
        only wraps the keyword arguments that are used in most common models.

        Parameters
        ------------------------
        **kwargs: Dict,
            Dictionary of keyword arguments to pass to parent class.
        """
        super().__init__(**kwargs)

    def _space(self) -> Dict:
        """Return space of hyper-parameters of the layer."""
        return {}

    def _build(self, input_layers: Layer, **kwargs) -> Layer:
        """Build Flatten layer.
        
        Parameters
        --------------------------
        input_layers: Layer,
            The input layer of the current layer.

        Returns
        --------------------------
        Built flatten layer.
        """
        return Flatten()(input_layers)
