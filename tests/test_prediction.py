import numpy as np

from RACER import RACER


def _configured_racer(fallback="majority"):
    racer = RACER(fallback=fallback, suppress_warnings=True)
    racer._has_fit = True
    racer._final_rules_if = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
        ],
        dtype=bool,
    )
    racer._final_rules_then = np.array(
        [[1, 0], [0, 1], [1, 0]], dtype=bool
    )
    racer._fitnesses = np.array([0.7, 0.9, 0.9])
    racer._majority_then = np.array([1, 0], dtype=bool)
    return racer


def test_predict_with_zero_rules_uses_majority_without_name_error():
    racer = _configured_racer()
    racer._final_rules_if = np.empty((0, 4), dtype=bool)
    racer._final_rules_then = np.empty((0, 2), dtype=bool)
    racer._fitnesses = np.empty(0)

    predictions = racer.predict(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=bool),
        convert_dummies=False,
    )

    assert np.array_equal(predictions, np.array([[1, 0], [1, 0]], dtype=bool))


def test_majority_remains_the_default_uncovered_instance_fallback():
    racer = _configured_racer()

    prediction = racer.predict(
        np.array([[1, 1, 1, 0]], dtype=bool), convert_dummies=False
    )

    assert np.array_equal(prediction[0], racer._majority_then)


def test_partial_match_ties_by_fitness_then_rule_order():
    racer = _configured_racer(fallback="partial-match")

    prediction = racer.predict(
        np.array([[1, 1, 1, 0]], dtype=bool), convert_dummies=False
    )

    assert np.array_equal(prediction[0], racer._final_rules_then[1])


def test_confusion_docstring_matches_return_order():
    assert "Returns n_covered and n_correct" in RACER._confusion.__doc__
