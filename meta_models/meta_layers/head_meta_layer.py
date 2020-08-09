"""Class implementing meta-model for a Dense Layer."""
from typing import Dict, List
from tensorflow.keras.layers import Layer, Dense, Activation, BatchNormalization
from tensorflow.keras import regularizers
from .dense_meta_layer import DenseMetaLayer


class HeadMetaLayer(DenseMetaLayer):

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
            **kwargs
        )