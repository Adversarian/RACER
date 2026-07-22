import numpy as np
import pytest

from RACER import RACER


SMALL_BINARY_X = np.array(
    [
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ],
    dtype=bool,
)
SMALL_BINARY_Y = np.array(
    [[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]], dtype=bool
)

MULTIWORD_X = np.zeros((6, 70), dtype=bool)
MULTIWORD_X[0, [0, 65]] = True
MULTIWORD_X[1, [1, 65]] = True
MULTIWORD_X[2, [2, 66]] = True
MULTIWORD_X[3, [3, 66]] = True
MULTIWORD_X[4, [4, 69]] = True
MULTIWORD_X[5, [5, 69]] = True
MULTIWORD_Y = np.array(
    [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
    dtype=bool,
)


@pytest.mark.parametrize(
    ("X", "y"),
    [
        pytest.param(SMALL_BINARY_X, SMALL_BINARY_Y, id="single-word"),
        pytest.param(MULTIWORD_X, MULTIWORD_Y, id="multiple-words"),
    ],
)
def test_packed_coverage_is_bit_identical_to_boolean_reference(X, y):
    packed = RACER(alpha=0.9)
    reference = RACER(alpha=0.9)
    reference._use_packed_coverage = False

    packed.fit(X, y)
    reference.fit(X, y)

    assert packed._coverage_backend == "uint64"
    assert reference._coverage_backend == "boolean-reference"
    assert np.array_equal(packed._final_rules_if, reference._final_rules_if)
    assert np.array_equal(packed._final_rules_then, reference._final_rules_then)
    assert packed._fitnesses.tobytes() == reference._fitnesses.tobytes()
