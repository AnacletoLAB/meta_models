"""Module providing stratified random holdouts generator."""
from typing import Union, Generator, Tuple
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from tqdm.auto import tqdm
from sanitize_ml_labels import sanitize_ml_labels


def stratified_holdouts(
    train_size: float,
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int = 1,
    random_state: int = 42,
    task_name: str = "",
    verbose: bool = True,
    leave: bool = False
) -> Union[Generator, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Return generator for stratified random holdouts.

    Parameters
    -----------------------
    train_size: float,
        The rate of values to leave in the training set.
    X: pd.DataFrame,
        The input data.
    y: pd.DataFrame,
        The output data labels.
    n_splits: int = 1,
        Number of holdouts.
    random_state: int = 42,
        The random state to reproduce the sampled holdouts.
    task_name: str = "",
        Name of the task to be shown in the loading bar.
    verbose: bool = True,
        Wether to show the loading bar.
        By default, True.
    leave: bool = False,
        Wether to leave the loading bar.
        By default, False.

    Returns
    -----------------------
    Generator with stratified random holdouts.
    """
    iterator = (
        (
            holdout_number,
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx]
        )
        for holdout_number, (train_idx, test_idx) in tqdm(
            enumerate(StratifiedShuffleSplit(
                n_splits=n_splits,
                train_size=train_size,
                random_state=random_state
            ).split(X, y)),
            desc="Computing holdouts for task {}".format(
                sanitize_ml_labels(task_name)
            ),
            total=n_splits,
            disable=not verbose,
            leave=leave
        )
    )
    if n_splits == 1:
        return next(iterator)[1:]
    return iterator
