"""Class implementing meta-model for a Conv2D Residual Layer.

That is, a rectangle of Conv2D layers where the first layer is connected
to the last one. This solution reduces the gradient disappearence problem
that is present in deep neural networks.

"""
from typing import Dict

from tensorflow.keras.layers import Add, Layer

from .conv2d_meta_layer import Conv2DMetaLayer
from .maxpool2d_meta_layer import MaxPool2DMetaLayer
from ..utils import distributions


class Conv2DRectangularMetaLayer(Conv2DMetaLayer):
    """Class handling a rectangular block of bidimensional convolutional layers.

    The class handles, optionally, residuality between the first and last
    layer of the block using a addition layer.

    Private members
    ---------------------------
    _min_layers: int,
        Minimum number of layers in rectangle.
        If the tuning process passes 0, then the layer is skipped.
    _max_layers: int,
        Maximum number of layers in rectangle.
    _min_x_strides: int,
        Minimum stride for the last layer of the Conv2D block.
        This is the minimal stride considered for the horizontal axis.
    _max_x_strides: int,
        Maximum stride for the last layer of the Conv2D block.
        This is the maximal stride considered for the horizontal axis.
    _min_y_strides: int,
        Minimum stride for the last layer of the Conv2D block.
        This is the minimal stride considered for the vertical axis.
    _max_y_strides: int,
        Maximum stride for the last layer of the Conv2D block.
        This is the maximal stride considered for the vertical axis.
    _residual: bool,
        Whether to apply residuality, by summing the first layer to
        the last layer. This only is applied when the optimization process
        suggests to use more than two layers.
    """

    def __init__(
        self,
        min_layers: int = 0,
        max_layers: int = 3,
        min_x_strides: int = 1,
        max_x_strides: int = 4,
        min_y_strides: int = 1,
        max_y_strides: int = 4,
        min_x_pool_size: int = 1,
        max_x_pool_size: int = 8,
        min_y_pool_size: int = 1,
        max_y_pool_size: int = 4,
        strides: bool = False,
        max_pooling: bool = True,
        residual: bool = False,
        **kwargs: Dict
    ):
        """Create new Conv2DResidualLayer meta-model object.

        Parameters
        ----------------------
        min_layers: int = 0,
            Minimum number of layers in rectangle.
            If the tuning process passes 0, then the layer is skipped.
        max_layers: int = 3,
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
        min_x_pool_size: int = 1,
            Minimum value of pool size.
            If the tuning process passes 0, then the maxpooling layer is skipped.
        max_x_pool_size: int = 8,
            Maximum value of pool size.
            If the tuning process passes 0, then the maxpooling layer is skipped.
        min_y_pool_size: int = 1,
            Minimum value of pool size.
            If the tuning process passes 0, then the maxpooling layer is skipped.
        max_y_pool_size: int = 4,
            Maximum value of pool size.
            If the tuning process passes 0, then the maxpooling layer is skipped.
        strides: bool = False,
            Wether to enable the strides.
        max_pooling: bool = True,
            Wether to enable the maxpooling layer.
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
        if not strides:
            min_x_strides = max_x_strides = 1
            min_y_strides = max_y_strides = 1
        self._min_x_strides = min_x_strides
        self._max_x_strides = max_x_strides
        self._min_y_strides = min_y_strides
        self._max_y_strides = max_y_strides
        self._max_pooling = MaxPool2DMetaLayer(
            min_x_pool_size=min_x_pool_size,
            max_x_pool_size=max_x_pool_size,
            min_y_pool_size=min_y_pool_size,
            max_y_pool_size=max_y_pool_size,
            enabled=max_pooling
        )

    def _space(self) -> Dict:
        """Return hyper parameters of the layer."""
        return {
            **super()._space(),
            "x_strides": (distributions.integer, self._min_x_strides, self._max_x_strides),
            "y_strides": (distributions.integer, self._min_y_strides, self._max_y_strides),
            "layers": (distributions.integer, self._min_layers, self._max_layers),
            **self._max_pooling._space()
        }

    def _build(
        self,
        input_layers: Layer,
        layers: int,
        x_strides: int,
        y_strides: int,
        x_pool_size: int,
        y_pool_size: int,
        dropout_rate: float,
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
        x_pool_size: float,
            The rate of pool size for the x axis.
            If the value is 0, the layer is not added.
        y_pool_size: float,
            The rate of pool size for the y axis.
            If the value is 0, the layer is not added.
        dropout_rate: float,
            The rate of dropout.
            If the value is very close to 0, the layer is not added.
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
        last = self._max_pooling._build(
            x_pool_size=x_pool_size,
            y_pool_size=y_pool_size,
            input_layers=last
        )
        return self._dropout._build(
            dropout_rate=dropout_rate,
            input_layers=last,
            **kwargs
        )
