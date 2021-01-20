from typing import Dict
import numpy as np


def random_space_sampling(space: Dict) -> Dict:
    """Return random space uniform sample."""
    return {
        key: np.random.uniform(low, high)
        for key, (_, low, high) in space.items()
    }
