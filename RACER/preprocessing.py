from typing import Tuple, Union

import numpy as np
import pandas as pd

from optbinning import MDLP, MulticlassOptimalBinning as MOB
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


class RACERPreprocessor:
    def __init__(self, target: str = "auto", max_n_bins=32, max_num_splits=32):
        """RACER preprocessing step that quantizes numerical columns and dummy encodes the categorical ones.
        Quantization is based on the optimal binning algorithm for "multiclass" tasks and the entropy-based MDLP
        algorithm for "binary" tasks.

        Args:
            target (str, optional): Whether the task is "multiclass" or "binary" classification. Defaults to "auto" which attempts automatically infer the task from `y`.
            max_n_bins (int, optional): Maximum number of bins to quantize in. Defaults to 32.
            max_num_splits (int, optional): Maximum number of splits to consider at each partition for MDLP. Defaults to 32.
        """
        assert target in [
            "multiclass",
            "binary",
            "auto",
        ], "`target` must either be 'multiclass', 'binary' or 'auto'."
        if target == "multiclass":
            self._quantizer = MOB(max_n_bins=max_n_bins)
        elif target == "binary":
            self._quantizer = MDLP(max_candidates=max_num_splits)
        else:
            self._quantizer = "infer"
            self._max_n_bins = max_n_bins
            self._max_candidates = max_num_splits

    def fit_transform_pandas(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocesses the dataset by replacing nominal vaues with dummy variables.
        Converts to numpy boolean arrays and returns the dataset. All numerical values are discretized
        using an optimal binning strategy that employs a decision tree as a preprocessing step.
        (This uses the legacy pandas dummy encoder. You can use this to retain total backward compatibility with previous code)

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features matrix
            y (Union[pd.DataFrame, np.ndarray]): Targets vector

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed features matrix and targets vectors.
        """
        X, y = pd.DataFrame(X), pd.DataFrame(y)
        if self._quantizer == "infer":
            uniques = y.nunique().values
            if uniques > 2:
                self._quantizer = MOB(max_n_bins=self._max_n_bins)
            else:
                self._quantizer = MDLP(max_candidates=self._max_candidates)
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

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocesses the dataset by replacing nominal vaues with dummy variables.
        Converts to numpy boolean arrays and returns the dataset. All numerical values are discretized
        using an optimal binning strategy that employs a decision tree as a preprocessing step.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features matrix
            y (Union[pd.DataFrame, np.ndarray]): Targets vector

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed features matrix and targets vectors.
        """
        X, y = pd.DataFrame(X), pd.DataFrame(y)
        if self._quantizer == "infer":
            uniques = y.nunique().values
            if uniques > 2:
                self._quantizer = MOB(max_n_bins=self._max_n_bins)
            else:
                self._quantizer = MDLP(max_candidates=self._max_candidates)
        numerics_X = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerics_X:
            for col in numerics_X:
                self._quantizer.fit(X[col].values, np.squeeze(y.values))
                bins = [X[col].min()] + self._quantizer.splits.tolist() + [X[col].max()]
                X[col] = pd.cut(X[col], bins=bins, include_lowest=True, labels=False)
        X, y = X.astype("category"), y.astype("category")
        X = OneHotEncoder(sparse_output=False).fit_transform(X).astype(bool)
        y = LabelBinarizer().fit_transform(y).astype(bool)
        return X, y

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ):
        """Fits the preprocessor on X and y for downstream transformations.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features vector
            y (Union[pd.DataFrame, np.ndarray]): Targets vector
        """
        print(
            "It is strongly recommended that you use fit_transform on your entire dataset."
        )
        print(
            "Use this option ONLY if you're certain new unseen values will not be encountered at test time."
        )
        X, y = pd.DataFrame(X), pd.DataFrame(y)
        if self._quantizer == "infer":
            uniques = y.nunique().values
            if uniques > 2:
                self._quantizer = MOB(max_n_bins=self._max_n_bins)
            else:
                self._quantizer = MDLP(max_candidates=self._max_candidates)
        numerics_X = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerics_X:
            self._bins = []
            for col in numerics_X:
                self._quantizer.fit(X[col].values, np.squeeze(y.values))
                bins = [X[col].min()] + self._quantizer.splits.tolist() + [X[col].max()]
                self._bins.append(bins)
                X[col] = pd.cut(X[col], bins=bins, include_lowest=True, labels=False)
        X, y = X.astype("category"), y.astype("category")
        self._X_encoder = OneHotEncoder(sparse_output=False).fit(X)
        self._y_encoder = LabelBinarizer().fit(y)

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms the provided new X and y with previously fitted preprocessor.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features matrix
            y (Union[pd.DataFrame, np.ndarray]): Targets vector

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed features matrix and targets vectors.
        """
        X, y = pd.DataFrame(X), pd.DataFrame(y)
        numerics_X = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerics_X:
            for col, bin in zip(numerics_X, self._bins):
                X[col] = pd.cut(X[col], bins=bin, include_lowest=True, labels=False)
        X, y = X.astype("category"), y.astype("category")
        X, y = self._X_encoder.transform(X).astype(bool), self._y_encoder.transform(
            y
        ).astype(bool)
        return X, y
