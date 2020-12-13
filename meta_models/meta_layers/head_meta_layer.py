"""Class implementing meta-model for a Dense Layer."""
from typing import Dict

from .dense_meta_layer import DenseMetaLayer


class HeadMetaLayer(DenseMetaLayer):
    """Class implementing meta-layer for Head layers.

    The Head meta-layer is a single neuron Dense layer with sigmoid actication
    that is meant to be the head layer of a classifier model. This layer can be
    customized to be used with multiple output classes by changing the activation
    from a sigmoid to a softmax.
    """

    def __init__(
        self,
        units: int = 1,
        activation: str = "sigmoid",
        **kwargs: Dict
    ):
        """Create new DenseResidualLayer meta-model object.

        Parameters
        ----------------------
        units: int = 1,
            Number of units (neurons) in each layer.
        activation: str = "sigmoid",
            The activation function to use for the layer.
        **kwargs: Dict,
            Dictionary of keyword parameters to be passed to parent class.
        """
        super().__init__(
            min_units=units,
            max_units=units,
            activation=activation,
            **kwargs
        )
