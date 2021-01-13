import numpy as np
from meta_models.meta_models import (CNN1DMetaModel, CNN2DMetaModel,
                                     CNN3DMetaModel, FFNNMetaModel)
from tqdm.auto import tqdm, trange

from .utils import random_space_sampling


def test_meta_model():
    """Testing if we can build and run the meta-models."""
    models = (
        FFNNMetaModel,
        CNN1DMetaModel,
        CNN2DMetaModel,
        CNN3DMetaModel
    )
    shapes = (
        (10,),
        (10, 5),
        (10, 5, 2),
        (10, 5, 4, 2)
    )
    FUZZYING_ITERATIONS = 5
    for model, shape in tqdm(
        zip(models, shapes),
        total=len(shapes),
        desc="Testing models"
    ):
        meta_model = model(
            shape,
            meta_layer_kwargs=dict(
                batch_normalization=True,
                activity_regularizer=True,
                kernel_regularizer=True,
                bias_regularizer=True
            )
        )
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
                np.random.uniform(0, 1, size=(100, *shape)),
                np.random.randint(2, size=(100,)),
                verbose=False
            )
