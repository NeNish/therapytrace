"""
scikit-learn transformers shared between training and inference.

These live in the application package rather than in the training script for a
practical reason: a pickled sklearn Pipeline stores the *import path* of every
custom transformer it contains. If they were defined in `train.py`, the saved
model would only load in a process that happened to have that script on its
path — which is fine on the training machine and broken everywhere else.

Defining them here means `joblib.load` works from the API, from a test, from a
notebook, and from a container that never ships the training code.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .features import AUXILIARY, DIMENSIONS, score_utterance


class Column(BaseEstimator, TransformerMixin):
    """Selects one text column, so a FeatureUnion can operate on a DataFrame."""

    def __init__(self, name: str = "text"):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.name].astype(str).values


class LexiconFeatures(BaseEstimator, TransformerMixin):
    """
    Turns each utterance into the TherapyTrace process scores.

    Ten interpretable features — the five process dimensions plus five
    auxiliaries — followed by length and a question flag. This is the bridge
    between the rule-based measurement layer and the supervised models: it lets
    an ablation ask directly whether the lexicon carries signal that bag-of-words
    does not.
    """

    FEATURES = list(DIMENSIONS) + list(AUXILIARY)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            s = score_utterance(str(text))
            rows.append(
                [getattr(s, f) for f in self.FEATURES]
                + [min(1.0, s.n_words / 60.0), float("?" in str(text))]
            )
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.FEATURES + ["length", "has_question"])
