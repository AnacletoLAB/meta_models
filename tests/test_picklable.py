import numpy as np
from meta_models.meta_models import (
    CNN2DMetaModel,
    FFNNMetaModel,
    MMNNMetaModel
)
from tqdm.auto import tqdm, trange
import pickle
from .utils import random_space_sampling


def test_picklable():
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

    space = meta_model.space()
    pickle.dumps(meta_model)
    model = meta_model.build(**random_space_sampling(space))
    pickle.dumps(meta_model)
