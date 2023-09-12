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

def test_dominance_independence():
    X, y = load_iris(return_X_y=True)
    X, y = RACERPreprocessor().fit_transform(X, y)
    racer = RACER(alpha=0.95)
    racer.fit(X, y)
    for i in range(len(racer._final_rules_if)):
        for j in range(i+1, len(racer._final_rules_if)):
            assert not racer._covered(
                racer._final_rules_if[j], racer._final_rules_if[i]
            )