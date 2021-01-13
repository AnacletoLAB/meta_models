"""Class implementing meta-model for a Conv3D Residual Layer.

That is, a rectangle of Conv3D layers where the first layer is connected
to the last one. This solution reduces the gradient disappearence problem
that is present in deep neural networks.

"""
from typing import Dict

from tensorflow.keras.layers import Add, Layer

from .conv3d_meta_layer import Conv3DMetaLayer


class Conv3DRectangularMetaLayer(Conv3DMetaLayer):
    """Class handling a rectangular block of tridimensional convolutional layers.

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
        Minimum stride for the last layer of the Conv3D block.
        This is the minimal stride considered for the length axis.
    _max_x_strides: int,
        Maximum stride for the last layer of the Conv3D block.
        This is the maximal stride considered for the length axis.
    _min_y_strides: int,
        Minimum stride for the last layer of the Conv3D block.
        This is the minimal stride considered for the depth axis.
    _max_y_strides: int,
        Maximum stride for the last layer of the Conv3D block.
        This is the maximal stride considered for the depth axis.
    _min_y_strides: int,
        Minimum stride for the last layer of the Conv3D block.
        This is the minimal stride considered for the height axis.
    _max_y_strides: int,
        Maximum stride for the last layer of the Conv3D block.
        This is the maximal stride considered for the height axis.
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
        min_z_strides: int = 1,
        max_z_strides: int = 4,
        residual: bool = False,
        **kwargs: Dict
    ):
        """Create new Conv3DResidualLayer meta-model object.

        Parameters
        ----------------------
        min_layers: int = 0,
            Minimum number of layers in rectangle.
            If the tuning process passes 0, then the layer is skipped.
        max_layers: int = 3,
            Maximum number of layers in rectangle.
        min_x_strides: int = 1,
            Minimum stride for the last layer of the Conv3D block.
            This is the minimal stride considered for the length axis.
        max_x_strides: int = 4,
            Maximum stride for the last layer of the Conv3D block.
            This is the maximal stride considered for the length axis.
        min_y_strides: int = 1,
            Minimum stride for the last layer of the Conv3D block.
            This is the minimal stride considered for the depth axis.
        max_y_strides: int = 4,
            Maximum stride for the last layer of the Conv3D block.
            This is the maximal stride considered for the depth axis.
        min_y_strides: int = 1,
            Minimum stride for the last layer of the Conv3D block.
            This is the minimal stride considered for the height axis.
        max_y_strides: int = 4,
            Maximum stride for the last layer of the Conv3D block.
            This is the maximal stride considered for the height axis.
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
        self._min_z_strides = min_z_strides
        self._max_z_strides = max_z_strides

    def _space(self) -> Dict:
        """Return hyper parameters of the layer."""
        return {
            **super()._space(),
            "x_strides": (self._min_x_strides, self._max_x_strides),
            "y_strides": (self._min_y_strides, self._max_y_strides),
            "z_strides": (self._min_z_strides, self._max_z_strides),
            "layers": (self._min_layers, self._max_layers)
        }

    def _build(
        self,
        input_layers: Layer,
        layers: int,
        x_strides: int,
        y_strides: int,
        z_strides: int,
        **kwargs
    ) -> Layer:
        """Return built Conv3D Residual layer block.

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
            This is the stride considered for the length axis.
        y_strides: int,
            The strides to use for the last layer of the block.
            This is the stride considered for the depth axis.
        z_strides: int,
            The strides to use for the last layer of the block.
            This is the stride considered for the hight axis.
        **kwargs: Dict,
            The kwargs to pass to the kernel regularizers.

        Returns
        --------------------------
        Output layer of the block.
        """
        layers = round(layers)
        x_strides = round(x_strides)
        y_strides = round(y_strides)
        z_strides = round(z_strides)
        strides = (x_strides, y_strides, z_strides)
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
