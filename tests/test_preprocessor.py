import numpy as np
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
