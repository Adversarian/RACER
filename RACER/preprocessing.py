from typing import Tuple, Union

import numpy as np
import pandas as pd

from optbinning import MDLP


class RACERPreprocessor:
    def __init__(self, max_num_splits=32):
        """RACER preprocessing step that quantizes numerical columns and dummy encodes the categorical ones.
        Quantization is based on the entropy-based  Minimum Description Length Principle algorithm (MDLP).

        Args:
            max_num_splits (int, optional): Maximum number of splits to consider at each partition for MDLP. Defaults to 32.
        """
        self._max_num_splits = max_num_splits

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]:
        """Preprocesses the dataset by replacing nominal vaues with dummy variables.
        Converts to numpy boolean arrays and returns the dataset. All numerical values are discretized
        using an optimal binning strategy that employs a decision tree as a preprocessing step.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features vector
            y (Union[pd.DataFrame, np.ndarray]): Targets vector

        Returns:
            Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]: Transformed features and targets vectors.
        """
        X, y = pd.DataFrame(X), pd.DataFrame(y)
        numerics_X = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerics_X:
            for col in numerics_X:
                optb = MDLP(max_candidates=self._max_num_splits)
                optb.fit(X[col].values, np.squeeze(y.values))
                X[col] = pd.cut(X[col], bins=optb.splits, duplicates="drop")
        X, y = X.astype("category"), y.astype("category")
        X = pd.get_dummies(X).to_numpy()
        y = pd.get_dummies(y).to_numpy()
        return X, y

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ):
        raise NotImplementedError(
            "Applying transformation across different datasets is not currently supported. Please use fit_transform instead."
        )

    def transform(self):
        raise NotImplementedError(
            "Applying transformation across different datasets is not currently supported. Please use fit_transform instead."
        )
