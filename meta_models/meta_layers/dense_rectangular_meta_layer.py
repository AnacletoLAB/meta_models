"""Class implementing meta-model for a Dense Residual Layer.

That is, a rectangle of dense layers where the first layer is connected
to the last one. This solution reduces the gradient disappearence problem
that is present in deep neural networks.

"""
from typing import Dict
from tensorflow.keras.layers import Layer, Add
from .dense_meta_layer import DenseMetaLayer


class DenseRectangularMetaLayer(DenseMetaLayer):
    """Class implementing meta-layer for Rectangular Dense layers.

    Private members
    ------------------------
    _min_layers: int,
        Minimum number of layers in rectangle.
        If the tuning process passes 0, then the layer is skipped.
    _max_layers: int,
        Maximum number of layers in rectangle.
    _residual: bool,
        Whether to apply residuality, by summing the first layer to
        the last layer. This only is applied when the optimization process
        suggests to use more than two layers.
    """

    def __init__(
        self,
        min_layers: int = 0,
        max_layers: int = 5,
        residual: bool = False,
        **kwargs: Dict
    ):
        """Create new DenseResidualLayer meta-model object.

        Parameters
        ----------------------
        min_layers: int = 0,
            Minimum number of layers in rectangle.
            If the tuning process passes 0, then the layer is skipped.
        max_layers: int = 5,
            Maximum number of layers in rectangle.
        residual: bool = False,
            Whether to apply residuality, by summing the first layer to
            the last layer. This only is applied when the optimization process
            suggests to use more than two layers.
        **kwargs: Dict,
            Dictionary of keyword parameters to be passed to parent class.
        """
        super().__init__(**kwargs)
        self._min_layers = min_layers
        self._max_layers = max_layers
        self._residual = residual

    def _space(self) -> Dict:
        """Return hyper parameters of the layer."""
        return {
            **super()._space(),
            "layers": (self._min_layers, self._max_layers)
        }

    def _build(
        self,
        input_layers: Layer,
        units: int,
        layers: int,
        **kwargs: Dict
    ) -> Layer:
        """Return built Dense Residual layer block.

        If the given layers number is equal to 0, the layer is skipped.

        Parameters
        --------------------------
        input_layers: Layer,
            The input layer of the current layer.
        units: int,
            The number of neurons of the layer.
        layers: int,
            The number of layers of the block.
        **kwargs: Dict,
            The kwargs to pass to the kernel regularizers.

        Returns
        --------------------------
        Output layer of the block.
        """
        layers = round(layers)
        # If no layer has been requested, we return the provided input
        if layers == 0:
            return input_layers
        # Otherwise we create the first layer
        hidden = first = super()._build(input_layers, units, **kwargs)
        # And add on top all the requested layers minus one
        for _ in range(1, layers-1):
            hidden = super()._build(hidden, units, **kwargs)
        # Finally, we add the last layer with residual sum when at least
        # 2 layers have been requested.
        last = hidden if layers <= 2 else super()._build(
            Add()([first, hidden]) if self._residual else hidden,
            units,
            **kwargs
        )
        return last
