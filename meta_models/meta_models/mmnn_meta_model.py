"""Abstract class implementing FFNN MetaModel."""
from typing import Dict, Tuple, Union, List

from ..meta_layers import HeadMetaLayer, InputMetaLayer, ConcatenateMetaLayer
from .meta_model import MetaModel


class MMNNMetaModel(MetaModel):
    """Class implementing a Multi Modal FFNN."""

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
        return {}

    def _structure(self, input_layer: InputMetaLayer = None):
        """Create structure of the model."""
        input_layers, hidden_layers = list(zip(*[
            model._structure()
            for model in self._input_models
        ]))

        concatenation = ConcatenateMetaLayer()(hidden_layers)
        _, output_layers = self._output_model._structure(concatenation)

        return input_layers, output_layers
