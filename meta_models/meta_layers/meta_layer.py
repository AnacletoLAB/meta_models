"""Abstract class implementing abstract factory pattern for building layers."""
from typing import Union, List, Dict
import numpy as np
from tensorflow.keras.layers import Layer
from collections import ChainMap


class MetaLayer:
    """Abstract class implementing abstract factory for building Layer objects.

    Private members
    -----------------------------
    _input_layers: List[Layer],
        List of preceeding MetaLayers.
    """

    layer_ids = {}

    def __init__(self, separator: str = "_"):
        """Create new MetaLayer object.

        Parameters
        -------------------
        separator: str = "_",
            The separator to use for the hyper-parameters.
        """
        self._input_layers: List[MetaLayer] = []
        self._rendered_space = None
        self._rendered_defaults = None
        self.reset()
        self._separator = separator
        self._id = MetaLayer.layer_ids.get(self.__class__.__name__, 0)
        MetaLayer.layer_ids[self.__class__.__name__] = self._id + 1

    def __call__(self, input_layers: Union["MetaLayer", List["MetaLayer"]]) -> "MetaLayer":
        """Handles layers graph."""
        if isinstance(input_layers, MetaLayer):
            input_layers = [input_layers]
        self._input_layers = input_layers
        return self

    @property
    def layer_prefix(self) -> str:
        """Return the prefix used for the current instance of the layer."""
        return self._separator.join((self.__class__.__name__, str(self._id)))

    def reset(self):
        """Restore layer to pre-built status."""
        for layer in self._input_layers:
            layer.reset() 
        self._layer = None

    def _filter_relevant_kwargs(self, **kwargs: Dict) -> Dict:
        """Return kwargs relevant to current layer instance.

        Parameters
        --------------------
        **kwargs: Dict,
            Kwargs to be filtered.

        Returns
        --------------------
        Filtered kwargs.
        """
        layer_prefix = self.layer_prefix
        return {
            key[len(layer_prefix) + len(self._separator):]: value
            for key, value in kwargs.items()
            if key.startswith(layer_prefix)
        }

    def _space(self) -> Dict:
        """Return hyper-parameters space for the layer.

        Raises
        ------------------------
        NotImplementedError,
            When method is not properly overrided in child classes.
        """
        raise NotImplementedError(
            "Method _space must be implemented in child classes."
        )

    def space(self) -> Dict:
        """Return hyper-parameters space for this layer and previous ones.

        Returns
        -------------------------
        Dictionary with hyper parameters for 
        """
        if self._rendered_space is None:
            layer_space = {
                self._separator.join((self.layer_prefix, key)): value
                for key, value in self._space().items()
            }
            space = ChainMap(*[
                layer.space() for layer in self._input_layers
            ], layer_space)
            self._rendered_defaults = {
                key: first
                for key, (first, second) in layer_space.items()
                if np.isclose(first, second)
            }
            self._rendered_space = {
                key: value
                for key, value in space.items()
                if key not in self._rendered_defaults
            }
        return self._rendered_space

    def _build_previous(self, **kwargs: Dict) -> Union[Layer, List[Layer]]:
        """Return build previous layers.

        If there is only a single previous layer, the method returns only
        that layer, otherwise if there are no previous layers the method
        returns None. Finally, if there are multiple previous layers, the
        method returns a list of the previous layers.

        Parameters
        -----------------------
        **kwargs: Dict,
            Kwargs to pass to previous layers for building.

        Returns
        -----------------------
        Previous layers built.
        """
        layers = [
            layer.build(**kwargs)
            for layer in self._input_layers
        ]
        return (
            layers[0]
            if len(layers) == 1
            else None
            if len(layers) == 0
            else layers
        )

    def _build(self, input_layers: List[Layer] = None, **kwargs) -> Layer:
        """Return build layer with given kwargs.

        Raises
        --------------------------
        NotImplementedError,
            When method is not properly implemented in child class.
        """
        raise NotImplementedError(
            "Method _build must be implemented in child classes."
        )

    def build(self, **kwargs) -> Layer:
        """Return build layer with given kwargs."""
        if self._layer is None:
            self._layer = self._build(
                input_layers=self._build_previous(**kwargs),
                **self._filter_relevant_kwargs(
                    **kwargs,
                    **self._rendered_defaults
                )
            )
        return self._layer
