from meta_models.meta_models import (
    FFNNMetaModel,
    CNN1DMetaModel,
    CNN2DMetaModel,
    CNN3DMetaModel
)
from .utils import random_space_sampling
from tqdm.auto import trange, tqdm
import numpy as np


def test_meta_model():
    """Testing if """
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
    for model, shape in tqdm(
        zip(models, shapes),
        total=len(shapes),
        desc="Testing models"
    ):
        meta_model = model(shape)
        space = meta_model.space()
        for _ in trange(10, desc="Fuzzying model combinations", leave=False):
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
