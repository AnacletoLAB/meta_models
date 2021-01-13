"""Abstract class implementing FFNN MetaModel."""
from collections import ChainMap
from typing import Dict, List, Tuple

from ..meta_layers import ConcatenateMetaLayer, InputMetaLayer, MetaLayer
from .meta_model import MetaModel


class MMNNMetaModel(MetaModel):
    """Class implementing a Multi Modal FFNN.

    The class implements a meta-model for a multi-modal neural network.

    Private members
    --------------------------
    _input_models: List[MetaModel],
        The input models for the MMNN.
    _output_model: MetaModel,
        The output model of the MMNN.
    """

    def __init__(
        self,
        input_models: List[MetaModel],
        output_model: MetaModel
    ):
        """Create new MMNNMetaModel object.

        Parameters
        -----------------------
        input_models: List[MetaModel],
            The input models for the MMNN.
        output_model: MetaModel,
            The output model of the MMNN.
        """
        self._input_models = input_models
        self._output_model = output_model
        super().__init__()

    def _space(self) -> Dict:
        """Return hyper-parameters space for the model."""
        return {
            **ChainMap(*[
                input_model._space()
                for input_model in self._input_models
            ]),
            **self._output_model._space()
        }

    def structure(self, input_layer: InputMetaLayer = None) -> Tuple[List[MetaLayer]]:
        """Create structure of the model.

        Parameters
        -------------------
        input_layer: InputMetaLayer = None,
            The input layer for the structure.

        Returns
        -------------------
        Tuple of lists with input layers and output layers.
        """
        input_layers, hidden_layers = list(zip(*[
            model.structure()
            for model in self._input_models
        ]))

        concatenation = ConcatenateMetaLayer()(hidden_layers)
        _, output_layers = self._output_model.structure(concatenation)

        return input_layers, output_layers
