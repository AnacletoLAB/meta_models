from meta_models.meta_models import ResidualFFNNMetaModel
from .utils import random_space_sampling
from tqdm.auto import trange


def test_residual_ffnn_meta_model():
    meta_model = ResidualFFNNMetaModel(8)
    space = meta_model.space()
    print(space)
    model = meta_model.build(**random_space_sampling(space))
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy"
    )
    model.summary()
