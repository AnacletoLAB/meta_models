"""Class implementing meta-model for a Conv2D Residual Layer.

That is, a rectangle of Conv2D layers where the first layer is connected
to the last one. This solution reduces the gradient disappearence problem
that is present in deep neural networks.

"""
from typing import Dict
from tensorflow.keras.layers import Layer, Add
from .conv2d_meta_layer import Conv2DMetaLayer


class Conv2DRectangularMetaLayer(Conv2DMetaLayer):

    def __init__(
        self,
        min_layers: int = 0,
        max_layers: int = 5,
        min_x_strides: int = 1,
        max_x_strides: int = 4,
        min_y_strides: int = 1,
        max_y_strides: int = 4,
        residual: bool = False,
        **kwargs: Dict
    ):
        """Create new Conv2DResidualLayer meta-model object.

        Parameters
        ----------------------
        min_layers: int = 0,
            Minimum number of layers in rectangle.
            If the tuning process passes 0, then the layer is skipped.
        max_layers: int = 5,
            Maximum number of layers in rectangle.
        min_x_strides: int = 1,
            Minimum stride for the last layer of the Conv2D block.
            This is the minimal stride considered for the horizontal axis.
        max_x_strides: int = 4,
            Maximum stride for the last layer of the Conv2D block.
            This is the maximal stride considered for the horizontal axis.
        min_y_strides: int = 1,
            Minimum stride for the last layer of the Conv2D block.
            This is the minimal stride considered for the vertical axis.
        max_y_strides: int = 4,
            Maximum stride for the last layer of the Conv2D block.
            This is the maximal stride considered for the vertical axis.
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
        self._min_x_strides = min_x_strides
        self._max_x_strides = max_x_strides
        self._min_y_strides = min_y_strides
        self._max_y_strides = max_y_strides

    def _space(self) -> Dict:
        """Return hyper parameters of the layer."""
        return {
            **super()._space(),
            "x_strides": (self._min_x_strides, self._max_x_strides),
            "y_strides": (self._min_y_strides, self._max_y_strides),
            "layers": (self._min_layers, self._max_layers)
        }

    def _build(
        self,
        input_layers: Layer,
        layers: int,
        x_strides: int,
        y_strides: int,
        **kwargs
    ) -> Layer:
        """Return built Conv2D Residual layer block.

        If the given layers number is equal to 0, the layer is skipped.

        Parameters
        --------------------------
        input_layers: Layer,
            The input layer of the current layer.
        units: int,
            The number of neurons of the layer.
        layers: int,
            The number of layers of the block.
        x_strides: int,
            The strides to use for the last layer of the block.
            This is the stride considered for the horizontal axis.
        y_strides: int,
            The strides to use for the last layer of the block.
            This is the stride considered for the vertical axis.
        **kwargs: Dict,
            The kwargs to pass to the kernel regularizers.

        Returns
        --------------------------
        Output layer of the block.
        """
        layers = round(layers)
        x_strides = round(x_strides)
        y_strides = round(y_strides)
        strides = (x_strides, y_strides)
        # If no layer has been requested, we return the provided input
        if layers == 0:
            return input_layers
        # Otherwise we create the first layer
        hidden = first = super()._build(
            input_layers,
            **({} if layers > 1 else dict(strides=strides)),
            **kwargs
        )
        # And add on top all the requested layers minus one
        for _ in range(1, layers-1):
            hidden = super()._build(
                hidden,
                **({} if layers > 2 else dict(strides=strides)),
                **kwargs
            )
        # Finally, we add the last layer with residual sum when at least
        # 2 layers have been requested.
        last = hidden if layers <= 2 else super()._build(
            Add()([first, hidden]) if self._residual else hidden,
            strides=strides,
            **kwargs
        )
        return last
