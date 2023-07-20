from typing import Tuple, Union

import numpy as np
import pandas as pd

from optbinning import MulticlassOptimalBinning as MOB, MDLP


class RACERPreprocessor:
    def __init__(self, target: str="multiclass", max_n_bins=32, max_num_splits=32):
        """RACER preprocessing step that quantizes numerical columns and dummy encodes the categorical ones.
        Quantization is based on the optimal binning algorithm for "multiclass" tasks and the entropy-based MDLP
        algorithm for "binary" tasks.

        Args:
            target (str, optional): Whether the task is "multiclass" or "binary" classification. Defaults to "multiclass".
            max_n_bins (int, optional): Maximum number of bins to quantize in. Defaults to 32.
            max_num_splits (int, optional): Maximum number of splits to consider at each partition for MDLP. Defaults to 32.
        """
        assert target in ["multiclass", "binary"], "`target` must either be 'multiclass' or 'binary'"
        if target == "multiclass":
            self._quantizer = MOB(max_n_bins=max_n_bins)
        elif target == "binary":
            self._quantizer = MDLP(max_candidates=max_num_splits)

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
                self._quantizer.fit(X[col].values, np.squeeze(y.values))
                bins = [X[col].min()] + self._quantizer.splits.tolist() + [X[col].max()]
                X[col] = pd.cut(X[col], bins=bins, include_lowest=True, labels=False)
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
