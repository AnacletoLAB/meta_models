"""Abstract class implementing abstract factory pattern for building models."""
from collections import ChainMap
from typing import Dict, List, Tuple

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from ..meta_layers import MetaLayer


class MetaModel:
    """Abstract factory for building Models.

    The class handles the building of the meta-layers into a meta-model.

    Private members
    ---------------------------
    _inputs: List[MetaLayer],
        List of the meta layers to use for the inputs.
    _outputs: List[MetaLayer],
        List of the meta layers to use for the outputs.
    """

    def __init__(self, reset_layer_count: bool = True):
        """Create new MetaModel object.
        
        Parameters
        ----------------------------
        reset_layer_count: bool = True,
            Wether to reset the layer count when running build.
        """
        if reset_layer_count:
            MetaLayer.reset_counter()
        self._inputs, self._outputs = self.structure()
        if isinstance(self._inputs, MetaLayer):
            self._inputs = (self._inputs,)
        if isinstance(self._outputs, MetaLayer):
            self._outputs = (self._outputs,)

    def _space(self) -> Dict[str, Tuple]:
        """Return hyper-parameters space for the model.

        Raises
        ------------------------
        NotImplementedError,
            When method is not properly overrided in child classes.
        """
        raise NotImplementedError(
            "Method _space must be implemented in child classes."
        )

    def space(self) -> Dict[str, Tuple]:
        """Return hyper-parameters space for the model and its layers.

        Returns
        -------------------------
        Dictionary with hyper parameters for the model and its layers.
        """
        return ChainMap(*[
            layer.space() for layer in self._outputs
        ], self._space())

    def structure(self) -> Tuple[List[MetaLayer]]:
        """Build the structure of the meta_model."""
        raise NotImplementedError(
            "Method structure must be implemented in child classes."
        )

    def build(self, **kwargs: Dict) -> Model:
        """Create new model.

        Parameters
        ---------------------------
        **kwargs: Dict,
            Dictionary of kwargs to pass to the layers.

        Returns
        ---------------------------
        Built model using provided kwargs.
        """
        for layer in self._outputs:
            layer.reset()
        return Model(
            inputs=[
                layer.build(**kwargs)
                for layer in self._inputs
            ],
            outputs=[
                layer.build(**kwargs)
                for layer in self._outputs
            ]
        )
