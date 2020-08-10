from meta_models.meta_models import ResidualCNN1DMetaModel
from .utils import random_space_sampling
from tqdm.auto import trange


def test_residual_cnn1d_meta_model():
    meta_model = ResidualCNN1DMetaModel((200, 5))
    space = meta_model.space()
    for _ in trange(100):
        model = meta_model.build(**random_space_sampling(space))
        model.compile(
            optimizer="nadam",
            loss="binary_crossentropy"
        )
    model.summary()
