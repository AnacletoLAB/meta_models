"""Methods relative to subgpu training."""
from typing import List
import tensorflow as tf
from multiprocessing import cpu_count


def get_gpus() -> List["LogicalDevice"]:
    """Return list of detected GPUs."""
    return tf.config.experimental.list_physical_devices('GPU')


def enable_subgpu_training():
    """Enable subgpu training using tensorflow."""
    for gpu in get_gpus():
        tf.config.experimental.set_memory_growth(gpu, True)


def get_gpu_number() -> int:
    """Return number of available GPUs."""
    return len(get_gpus())


def get_minimum_gpu_rate_per_trial(process_number: int = None) -> float:
    """Return minimum number of GPU usage per trial."""
    if process_number is None:
        process_number = cpu_count()
    return get_gpu_number() / process_number
