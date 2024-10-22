"""Class implementing abstract Tuner."""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from extra_keras_metrics import (
    get_standard_binary_metrics,
    get_minimal_multiclass_metrics
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model

from ..meta_models import MetaModel
from ..utils import enable_subgpu_training


class Tuner:
    """Class implementing an abstract Tuner object.

    An abstract Tuner object tunes a meta-model and determines
    the near-optimal parameters of the model for the considered task.
    """

    def __init__(
        self,
        meta_model: MetaModel,
    ):
        """Create the Tuner object.

        Parameters
        --------------------------
        meta_model: MetaModel,
            The meta-model to optimize.
        """
        self._meta_model = meta_model
        self._model = None
        self._optimal_config = None

    def build(
        self,
        config: Dict = None,
        optimizer: str = "nadam",
        loss: str = "binary_crossentropy",
    ) -> Model:
        """Build model.

        Parameters
        ---------------------
        config: Dict = None,
            Selected hyper-parameters.
            If None is given, the stored optimal configuration is used.
        optimizer: str = "nadam",
            Optimizer to use for tuning.
        loss: str = "binary_crossentropy",
            Loss to use.

        Returns
        ----------------------
        Keras model.
        """
        if config is None:
            if self._optimal_config is None:
                raise ValueError(
                    "You must tune the hyper-parameters before running fit."
                )
            config = self._optimal_config
        # Build the selected model from the meta model
        self._model = self._meta_model.build(**config)
        # Compile it
        self._model.compile(
            optimizer=optimizer,
            loss=loss,
            # We add all the most common binary metrics
            metrics=(
                get_standard_binary_metrics()
                if loss == "binary_crossentropy"
                else get_minimal_multiclass_metrics()
            )
        )
        return self._model

    def fit(
        self,
        config: Dict = None,
        train: Tuple[np.ndarray] = None,
        validation_data: Tuple[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 256,
        monitor: str = "AUPRC",
        mode: str = "max",
        patience: int = 5,
        min_delta: float = 0.001,
        optimizer: str = "nadam",
        loss: str = "binary_crossentropy",
        callbacks: Tuple = (),
        verbose: bool = False,
        subgpu_training: bool = False
    ) -> pd.DataFrame:
        """Train the ray model.

        Parameters
        ---------------------
        config: Dict = None,
            Selected hyper-parameters.
        train: MixedSequence = None,
            Training sequence.
        validation_data: MixedSequence = None,
            Validation sequence.
        epochs: int = 100,
            Maximum number of training epochs.
        batch_size: int = 256,
            Batch size for the training process.
        patience: int = 5,
            Patience for early stopping.
        min_delta: float = 0.001,
            Minimum delta for early stopping.
        optimizer: str = "nadam",
            Optimizer to use for tuning.
        loss: str = "binary_crossentropy",
            Loss to use.
        callbacks: Tuple = (),
            Callbacks for the model.
        verbose: bool = False,
            Wether to show loading bars.
        enable_ray_callback: bool = True,
            Wether to enable the ray callback.
        subgpu_training: bool = False,
            Wether to enable subgpu training.

        Returns
        ----------------------
        Dataframe containing training history.
        """
        if subgpu_training:
            enable_subgpu_training()
        # Build the model
        self.build(
            config,
            optimizer,
            loss
        )
        # Fitting the model
        return pd.DataFrame(self._model.fit(
            *train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=True,
            callbacks=[
                *callbacks,
                # We kill the process when the training reaches a plateau
                EarlyStopping(
                    monitor=monitor,
                    mode=mode,
                    min_delta=min_delta,
                    patience=patience,
                    restore_best_weights=True
                )
            ]
        ).history)

    def evaluate(
        self,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """Train the ray model.

        Returns
        ----------------------
        Dataframe containing training history.
        """
        if self._model is None:
            raise ValueError("The model has not been fit!")
        # Fitting the model
        return dict(zip(
            self._model.metrics_names,
            self._model.evaluate(*args, **kwargs)
        ))

    def summary(self):
        """Print summary of the model."""
        if self._model is None:
            raise ValueError("The model has not been built!")
        self._model.summary()

    @property
    def optimal_configuration(self) -> Dict:
        """Return optimal configuration.

        Raises
        ----------------------
        ValueError,
            If the model has not been tuned yet.
        
        Returns
        ----------------------
        Dictionary with optimal configuration.
        """
        if self._optimal_config is None:
            raise ValueError("The model has not been tuned!")
        return self._optimal_config.copy()

    def tune(self) -> pd.DataFrame:
        """Tune the model.

        This method must be implemented in the child classes.
        """
        raise NotImplementedError(
            "Method tune must be implemented in child class."
        )
