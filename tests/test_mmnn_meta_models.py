import numpy as np
from meta_models.meta_models import (
    CNN2DMetaModel,
    FFNNMetaModel,
    MMNNMetaModel
)
from tqdm.auto import tqdm, trange

from .utils import random_space_sampling


def test_mmnn_meta_model():
    """Testing if we can build and run the MMNN meta-model."""
    cnn = CNN2DMetaModel(
        (200, 4, 1),
        input_name="cnn",
        meta_layer_kwargs=dict(
            batch_normalization=True
        ),
        top_ffnn_meta_model_kwargs=dict(
            meta_layer_kwargs=dict(
                batch_normalization=True
            ),
            headless=True
        )
    )
    mlp = FFNNMetaModel(10, input_name="mlp", headless=True)
    head = FFNNMetaModel()
    meta_model = MMNNMetaModel((mlp, cnn), head)

    FUZZYING_ITERATIONS = 10

    space = meta_model.space()
    for _ in trange(
        FUZZYING_ITERATIONS,
        desc="Fuzzying model combinations",
        leave=False
    ):
        model = meta_model.build(**random_space_sampling(space))
        model.compile(
            optimizer="nadam",
            loss="binary_crossentropy"
        )
        model.evaluate(
            {
                "cnn": np.random.uniform(0, 1, size=(100, 200, 4, 1)),
                "mlp": np.random.uniform(0, 1, size=(100, 10))
            },
            np.random.randint(2, size=(100,)),
            verbose=False
        )
