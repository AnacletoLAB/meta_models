from meta_models.meta_models import FFNNMetaModel
from .utils import random_space_sampling
from tqdm.auto import trange


def test_ffnn_meta_model():
    meta_model = FFNNMetaModel(8)
    space = meta_model.space()
    print(space)
    for _ in range(10):
        model = meta_model.build(**random_space_sampling(space))
        model.compile(
            optimizer="nadam",
            loss="binary_crossentropy"
        )
    model.summary()
