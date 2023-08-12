from RACER import RACER, RACERPreprocessor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def test_workflow_integrity():
    X, y = load_iris(return_X_y=True)
    X, y = RACERPreprocessor().fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, test_size=0.3
    )
    racer = RACER(alpha=0.95)
    racer.fit(X_train, y_train)
    acc = racer.score(X_test, y_test)
    assert acc > 0.75
