"""Abstract class implementing abstract factory pattern for building models."""
from collections import ChainMap
from typing import Dict, List, Tuple

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
        self._rendered = False
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
        self._rendered = True
        return dict(ChainMap(*[
            layer.space() for layer in self._outputs
        ], self._space()))

    def structure(self, input_layer: MetaLayer = None) -> Tuple[List[MetaLayer]]:
        """Create structure of the model.

        Parameters
        -------------------
        input_layer: InputMetaLayer = None,
            The input layer for the structure.

        Returns
        -------------------
        Tuple of lists with input layers and output layers.
        """
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
        if not self._rendered:
            self.space()
        model = Model(
            inputs=[
                layer.build(**kwargs)
                for layer in self._inputs
            ],
            outputs=[
                layer.build(**kwargs)
                for layer in self._outputs
            ]
        )
        for layer in self._outputs:
            layer.reset()
        return model
