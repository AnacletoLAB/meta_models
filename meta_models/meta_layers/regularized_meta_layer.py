"""Class implementing meta-model for a Dense Layer."""
from collections import ChainMap
from typing import Dict, List

import numpy as np
from tensorflow.keras import regularizers

from .meta_layer import MetaLayer


class RegularizedMetaLayer(MetaLayer):
    """Class implementing RegularizedMetaLayer.

    The class offers methods for handling the hyper-parameters
    related to l1 and l2 normalization (activity, kernel and bias) and offers
    the optional parameters to enable each of them singularly, other than
    additionally the batch normalization and the dropout.

    Private members
    -------------------------------
    _min_l1_regularization: float,
        Minimum value for L1 regularization.
    _max_l1_regularization: float,
        Maximum value for L1 regularization.
    _min_l2_regularization: float,
        Minimum value for L2 regularization.
    _max_l2_regularization: float,
        Maximum value for L2 regulatization,
    _min_dropout_rate: float,
        Minimum value for dropout rate.
    _max_dropout_rate: float,
        Maximum value for dropout rate.
    _batch_normalization: bool,
        Wethever to use a batch normalization.
    _activity_regularizer: bool,
        Wethever to use an activity regularizer.
    _kernel_regularizer: bool,
        Wethever to use an kernel regularizer.
    _bias_regularizer: bool,
        Wethever to use an bias regularizer.
    _dropout: bool,
        Wethever to use the dropout layer.
    """

    regularizers_types = ("l1", "l2")

    def __init__(
        self,
        l1_regularization: float = None,
        min_l1_regularization: float = 0,
        max_l1_regularization: float = 0.01,
        l2_regularization: float = None,
        min_l2_regularization: float = 0,
        max_l2_regularization: float = 0.01,
        min_dropout_rate: float = 0,
        max_dropout_rate: float = 0.5,
        batch_normalization: bool = False,
        activity_regularizer: bool = False,
        kernel_regularizer: bool = False,
        bias_regularizer: bool = False,
        dropout: bool = False,
        **kwargs: Dict
    ):
        """Create new DenseResidualLayer meta-model object.

        Parameters
        ----------------------
        l1_regularization: float = None,
            Exact value for l1 regularization.
            If provided, this automatically fixes the l1 norm parameter
            to the given value. You still need to enable which kind of
            normalization you inted to use.
            By default, this value is None, that is, disabled.
        min_l1_regularization: float = 0,
            Minimum value of l1 regularization.
            If the tuning process passes 0, then the regularization is skipped.
            This is the minimum value used for all form of regularization,
            but diffent keyword arguments are used depending on what is
            enabled in this layer, so different values will be passed by the
            optimization process. Rarily the regularization layers have vastly
            different values, hence the absence of multiple parameters.
        max_l1_regularization: float = 0.01,
            Maximum value of l1 regularization.
            This is the maximum value used for all form of regularization,
            but diffent keyword arguments are used depending on what is
            enabled in this layer, so different values will be passed by the
            optimization process. Rarily the regularization layers have vastly
            different values, hence the absence of multiple parameters.
        l2_regularization: float = None,
            Exact value for l2 regularization.
            If provided, this automatically enables L2 regularization and
            fixes the parameter to the given value.
            By default, this value is None, that is, disabled.
        min_l2_regularization: float = 0,
            Minimum value of l2 regularization.
            If the tuning process passes 0, then the regularization is skipped.
            This is the minimum value used for all form of regularization,
            but diffent keyword arguments are used depending on what is
            enabled in this layer, so different values will be passed by the
            optimization process. Rarily the regularization layers have vastly
            different values, hence the absence of multiple parameters.
        max_l2_regularization: float = 0.01,
            Maximum value of l2 regularization.
            This is the maximum value used for all form of regularization,
            but diffent keyword arguments are used depending on what is
            enabled in this layer, so different values will be passed by the
            optimization process. Rarily the regularization layers have vastly
            different values, hence the absence of multiple parameters.
        min_dropout_rate: float = 0,
            Minimum value of dropout.
            If the tuning process passes 0, then the dropout layer is skipped.
        max_dropout_rate: float = 0.5,
            Maximum value of dropout.
            If the tuning process passes 0, then the dropout layer is skipped.
        batch_normalization: bool = False,
            Wethever to use or not batch normalization.
        activity_regularizer: bool = False,
            Wethever to use an activity regularizer.
        kernel_regularizer: bool = False,
            Wethever to use a kernel regularizer.
        bias_regularizer: bool = False,
            Wethever to use a bias regularizer.
        dropout: bool = False,
            Wethever to use a dropout.
        **kwargs: Dict,
            Dictionary of keyword parameters to be passed to parent class.
        """
        super().__init__(**kwargs)
        if l1_regularization is not None:
            min_l1_regularization = max_l1_regularization = l1_regularization
        if l2_regularization is not None:
            min_l2_regularization = max_l2_regularization = l2_regularization
        self._min_l1_regularization = min_l1_regularization
        self._max_l1_regularization = max_l1_regularization
        self._min_l2_regularization = min_l2_regularization
        self._max_l2_regularization = max_l2_regularization
        self._min_dropout_rate = min_dropout_rate
        self._max_dropout_rate = max_dropout_rate
        self._batch_normalization = batch_normalization
        self._activity_regularizer = activity_regularizer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._dropout = dropout

    def _active_regularizer_types(self, regularizer: str) -> List[str]:
        """Return active regularizers."""
        return [
            "{}_{}".format(regularizer, reg_type)
            for reg_type in RegularizedMetaLayer.regularizers_types
        ]

    def _active_regularizers(self,) -> List[str]:
        """Return active regularizers."""
        return [
            regularizer
            for regularizer, enabled in zip(
                (
                    "activity_regularizer",
                    "kernel_regularizer",
                    "bias_regularizer"
                ),
                (
                    self._activity_regularizer,
                    self._kernel_regularizer,
                    self._bias_regularizer
                )
            )
            if enabled
        ]

    def _space(self) -> Dict:
        """Return hyper parameters of the layer."""
        return ChainMap(*[
            {
                regularizer_type: reg_range
                for regularizer_type, reg_range in zip(
                    self._active_regularizer_types(regularizer),
                    (
                        (
                            self._min_l1_regularization,
                            self._max_l1_regularization
                        ),
                        (
                            self._min_l2_regularization,
                            self._max_l2_regularization
                        )
                    )
                )
            }
            for regularizer in self._active_regularizers()
        ], {
            "dropout_rate": (self._min_dropout_rate, self._max_dropout_rate)
        } if self._dropout else {})

    def _build_regularizers(self, **kwargs: Dict) -> Dict:
        """Return regularizer for the current layer.

        Parameters
        -------------------------
        **kwargs:Dict,
            keyword arguments for the method, including, optionally:

            kernel_regularizer_l1: int,
                Weight to use for L1 kernel regularization.
            kernel_regularizer_l2: int,
                Weight to use for L2 kernel regularization.
            bias_regularizer_l1: int,
                Weight to use for L1 bias regularization.
            bias_regularizer_l2: int,
                Weight to use for L2 bias regularization.
            activity_regularizer_l1: int,
                Weight to use for L1 activity regularization.
            activity_regularizer_l2: int,
                Weight to use for L2 activity regularization.

        Returns
        ------------------------
        The regularizers keyword arguments and objects for the layer.
        """
        return {
            regularizer: regularizers.l1_l2(**{
                regularizer_type: kwargs[name]
                for regularizer_type, name in zip(
                    RegularizedMetaLayer.regularizers_types,
                    self._active_regularizer_types(regularizer)
                )
                if not np.isclose(kwargs[name], 0)
            })
            for regularizer in self._active_regularizers()
        }
