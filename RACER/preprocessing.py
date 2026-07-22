from typing import Tuple, Union

import numpy as np
import pandas as pd

from optbinning import MDLP, MulticlassOptimalBinning as MOB, OptimalBinning as OB
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


class InformationGainDiscretizer:
    """Single binary midpoint split from the RACER paper, Eqs. 1-3."""

    def fit(self, values: np.ndarray, y: np.ndarray):
        values = np.asarray(values, dtype=float).reshape(-1)
        y = np.asarray(y).reshape(-1)
        unique_values = np.unique(values)
        if len(unique_values) == 1:
            self.splits = np.array([], dtype=float)
            return self

        midpoints = (unique_values[:-1] + unique_values[1:]) / 2.0
        weighted_entropies = np.array(
            [self._weighted_entropy(values, y, midpoint) for midpoint in midpoints]
        )
        self.splits = np.array([midpoints[np.argmin(weighted_entropies)]])
        return self

    @staticmethod
    def _entropy(labels: np.ndarray) -> float:
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()
        return float(-(probabilities * np.log2(probabilities)).sum())

    @classmethod
    def _weighted_entropy(
        cls, values: np.ndarray, labels: np.ndarray, midpoint: float
    ) -> float:
        left = labels[values <= midpoint]
        right = labels[values > midpoint]
        total = len(labels)
        return (len(left) / total) * cls._entropy(left) + (
            len(right) / total
        ) * cls._entropy(right)


def _bin_edges(values: pd.Series, splits: np.ndarray) -> list:
    minimum, maximum = values.min(), values.max()
    if minimum == maximum:
        return [-np.inf, np.inf]
    return [minimum] + splits.tolist() + [maximum]


class RACERPreprocessor:
    def __init__(
        self,
        target: str = "auto",
        max_n_bins=32,
        max_num_splits=32,
        use_optimal_quantizer=False,
        discretizer="default",
    ):
        """RACER preprocessing step that quantizes numerical columns and dummy encodes the categorical ones.
        Quantization is based on the optimal binning algorithm for "multiclass" tasks and the entropy-based MDLP
        algorithm for "binary" tasks.

        Args:
            target (str, optional): Whether the task is "multiclass" or "binary" classification. Defaults to "auto" which attempts automatically infer the task from `y`.
            max_n_bins (int, optional): Maximum number of bins to quantize in. Defaults to 32.
            max_num_splits (int, optional): Maximum number of splits to consider at each partition for MDLP. Defaults to 32.
            discretizer (str, optional): ``"default"`` retains MDLP/optimal binning;
                ``"ig-paper"`` uses the original paper's single information-gain
                midpoint split. Defaults to ``"default"``.
        """
        assert target in [
            "multiclass",
            "binary",
            "auto",
        ], "`target` must either be 'multiclass', 'binary' or 'auto'."
        assert discretizer in [
            "default",
            "ig-paper",
        ], "`discretizer` must either be 'default' or 'ig-paper'."
        if discretizer == "ig-paper":
            self._quantizer = InformationGainDiscretizer()
        elif use_optimal_quantizer:
            self._quantizer = OB()
        else:
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
                bins = _bin_edges(X[col], self._quantizer.splits)
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
                bins = _bin_edges(X[col], self._quantizer.splits)
                X[col] = pd.cut(X[col], bins=bins, include_lowest=True, labels=False)
        X, y = X.astype("category"), y.astype("category")
        X = OneHotEncoder(sparse_output=False).fit_transform(X).astype(bool)
        y = LabelBinarizer().fit_transform(y).astype(bool)
        return X, y

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ):
        """Fits the preprocessor on training X and y for downstream transformations.

        Fit only on the training partition, then use :meth:`transform` for held-out
        data. This prevents target leakage through discretization and encoding.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features vector
            y (Union[pd.DataFrame, np.ndarray]): Targets vector
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
            self._bins = []
            for col in numerics_X:
                self._quantizer.fit(X[col].values, np.squeeze(y.values))
                bins = _bin_edges(X[col], self._quantizer.splits)
                self._bins.append(bins)
                X[col] = pd.cut(X[col], bins=bins, include_lowest=True, labels=False)
        X, y = X.astype("category"), y.astype("category")
        self._X_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        ).fit(X)
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
                clipped = X[col].clip(lower=bin[0], upper=bin[-1])
                X[col] = pd.cut(
                    clipped, bins=bin, include_lowest=True, labels=False
                )
        X, y = X.astype("category"), y.astype("category")
        X, y = self._X_encoder.transform(X).astype(bool), self._y_encoder.transform(
            y
        ).astype(bool)
        return X, y
