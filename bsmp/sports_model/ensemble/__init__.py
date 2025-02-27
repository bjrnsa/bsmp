"""This module contains ensemble methods for sports model predictions."""

from .nnls_ensemble import NNLS
from .simple_ensemble import SimpleEnsemble

__all__ = ["NNLS", "SimpleEnsemble"]
