import numpy as np
import pandas as pd
from RACER import RACERPreprocessor
from sklearn.datasets import load_iris


def test_output_equiv():
    X, y = load_iris(return_X_y=True)
    X1, y1 = RACERPreprocessor().fit_transform(X, y)
    X2, y2 = RACERPreprocessor(target="multiclass").fit_transform(X, y)
    X3, y3 = RACERPreprocessor().fit_transform_pandas(X, y)
    X4, y4 = RACERPreprocessor(target="multiclass").fit_transform_pandas(X, y)
    preprocessor1 = RACERPreprocessor()
    preprocessor1.fit(X, y)
    preprocessor2 = RACERPreprocessor(target="multiclass")
    preprocessor2.fit(X, y)
    X5, y5 = preprocessor1.transform(X, y)
    X6, y6 = preprocessor2.transform(X, y)
    Xs_tfmd, ys_tfmd = [X1, X2, X3, X4, X5, X6], [y1, y2, y3, y4, y5, y6]
    for i in range(1, len(Xs_tfmd)):
        assert np.all(Xs_tfmd[i] == Xs_tfmd[i - 1])
        assert np.all(ys_tfmd[i] == ys_tfmd[i - 1])
        assert Xs_tfmd[i].shape == Xs_tfmd[i - 1].shape
        assert ys_tfmd[i].shape == ys_tfmd[i - 1].shape


def test_transform_ignores_unseen_feature_categories():
    X_train = pd.DataFrame({"color": ["red", "blue", "red"]})
    y_train = np.array([0, 1, 0])
    preprocessor = RACERPreprocessor()
    preprocessor.fit(X_train, y_train)

    X_test = pd.DataFrame({"color": ["green", "red"]})
    X_transformed, _ = preprocessor.transform(X_test, np.array([0, 0]))

    assert not X_transformed[0].any()
    assert X_transformed[1].sum() == 1


def test_transform_clips_numeric_values_to_fitted_boundary_bins():
    X_train = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]})
    y_train = np.array([0, 0, 1, 1])
    preprocessor = RACERPreprocessor(target="binary")
    preprocessor.fit(X_train, y_train)

    boundary_X, _ = preprocessor.transform(
        pd.DataFrame({"value": [0.0, 3.0]}), np.array([0, 1])
    )
    outside_X, _ = preprocessor.transform(
        pd.DataFrame({"value": [-100.0, 100.0]}), np.array([0, 1])
    )

    assert np.array_equal(outside_X, boundary_X)
    assert np.all(outside_X.sum(axis=1) == 1)
