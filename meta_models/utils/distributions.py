"""Module providing enumeration of the possible tuples."""
from collections import namedtuple

distribution_names = [
    "real",
    "integer",
    "choice"
]

distributions = namedtuple(
    'Distributions',
    distribution_names
)(*distribution_names)

__all__ = [
    "distributions"
]
