from typing import Tuple

import numpy as np
from numpy import (
    bitwise_and as AND,
    bitwise_not as NOT,
    bitwise_or as OR,
    bitwise_xor as XOR,
)


def XNOR(input: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Computes the XNOR gate. (semantically the same as `input == other`)

    Args:
        input (np.ndarray): Input array
        other (np.ndarray): Other input array

    Returns:
        np.ndarray: XNOR(input, other) as an array
    """
    return NOT(XOR(input, other))


class RACER:
    def __init__(
        self,
        alpha=0.9,
        suppress_warnings=False,
        benchmark=False,
    ):
        """Initialize the RACER class

        Args:
            alpha (float, optional): Value of alpha according to the RACER paper. Defaults to 0.9.
            suppress_warnings (bool, optional): Whether to suppress any warnings raised during prediction. Defaults to False.
            benchmark (bool, optional): Whether to time the `fit` method for benchmark purposes. Defaults to False.
        """
        self._alpha, self._beta = alpha, 1.0 - alpha
        self._suppress_warnings = suppress_warnings
        self._benchmark = benchmark
        self._has_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the RACER algorithm on top of input data X and targets y.
        The code is written in close correlation to the pseudo-code provided in the RACER paper with some slight modifications.

        Args:
            X (np.ndarray): Features vector
            y (np.ndarray): Targets vector
        """
        if self._benchmark:
            from time import perf_counter

            tic = perf_counter()

        self._X, self._y = X, y
        self._cardinality, self._rule_len = self._X.shape
        self._classes = np.unique(self._y, axis=0)
        self._class_indices = {
            self._label_to_int(cls): np.where(XNOR(self._y, cls).min(axis=-1))[0]
            for cls in self._classes
        }

        self._create_init_rules()

        for cls in self._class_indices.keys():
            for i in range(len(self._class_indices[cls])):
                for j in range(i + 1, len(self._class_indices[cls])):
                    self._process_rules(
                        self._class_indices[cls][i], self._class_indices[cls][j]
                    )

        independent_indices = NOT(self._extants_covered)
        self._extants_if, self._extants_then, self._fitnesses = (
            self._extants_if[independent_indices],
            self._extants_then[independent_indices],
            self._fitnesses[independent_indices],
        )

        self._generalize_extants()

        # https://stackoverflow.com/questions/64238462/numpy-descending-stable-arg-sort-of-arrays-of-any-dtype
        args = (
            len(self._fitnesses)
            - 1
            - np.argsort(self._fitnesses[::-1], kind="stable")[::-1]
        )

        self._final_rules_if, self._final_rules_then, self._fitnesses = (
            self._extants_if[args],
            self._extants_then[args],
            self._fitnesses[args],
        )

        self._finalize_rules()

        self._has_fit = True

        if self._benchmark:
            self._bench_time = perf_counter() - tic

    def predict(self, X: np.ndarray, convert_dummies=True) -> np.ndarray:
        """Given input X, predict label using RACER

        Args:
            X (np.ndarray): Input features vector
            convert_dummies (bool, optional): Whether to convert dummy labels back to integert format. Defaults to True.

        Returns:
            np.ndarray: Label as predicted by RACER
        """
        assert self._has_fit, "RACER has not been fit yet."
        labels = np.zeros((len(X), self._final_rules_then.shape[1]), dtype=bool)
        found = np.zeros(len(X), dtype=bool)
        for i in range(len(self._final_rules_if)):
            covered = self._covered(X, self._final_rules_if[i])
            labels[AND(covered, NOT(found))] = self._final_rules_then[i]
            found[covered] = True
            all_found = found.sum() == len(X)
            if all_found:
                break

        if not all_found:
            if not self._suppress_warnings:
                print(
                    f"WARNING: RACER was unable to find a perfect match for {len(X) - found.sum()} instances out of {len(X)}"
                )
                print(
                    "These instances will be labelled as the majority class during training."
                )
            leftover_indices = np.where(NOT(found))[0]
            for idx in leftover_indices:
                labels[idx] = self._closest_match(X[idx])

        if convert_dummies:
            labels = np.argmax(labels, axis=-1)

        return labels

    def _bool2str(self, bool_arr: np.ndarray) -> str:
        """Converts a boolean array to a human-readable string

        Args:
            bool_arr (np.ndarray): The input boolean array

        Returns:
            str: Human-readable string output
        """
        return np.array2string(bool_arr.astype(int), separator="")

    def display_rules(self) -> None:
        """Print out the final rules"""
        assert self._has_fit, "RACER has not been fit yet."
        print("Algorithm Parameters:")
        print(f"\t- Alpha: {self._alpha}")
        if self._benchmark:
            print(f"\t- Time to fit: {self._bench_time}s")
        print(
            f"\nFinal Rules ({len(self._final_rules_if)} total): (if --> then (label) | fitness)"
        )
        for i in range(len(self._final_rules_if)):
            print(
                f"\t{self._bool2str(self._final_rules_if[i])} -->"
                f" {self._bool2str(self._final_rules_then[i])}"
                f" ({self._label_to_int(self._final_rules_then[i])})"
                f" | {self._fitnesses[i]}"
            )

    def _closest_match(self, X: np.ndarray) -> np.ndarray:
        """Find the closest matching rule to `X` (This will be extended later)

        Args:
            X (np.ndarray): Input rule `X`

        Returns:
            np.ndarray: Matched rule
        """
        return self._majority_then

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Returns accuracy on the provided test data.

        Args:
            X_test (np.ndarray): Test features vector
            y_test (np.ndarray): Test targets vector

        Returns:
            float: Accuracy score
        """
        assert self._has_fit, "RACER has not been fit yet."
        try:
            from sklearn.metrics import accuracy_score
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required to use the score function. Install wit `pip install scikit-learn`."
            )
        if y_test.ndim != 1 and y_test.shape[1] != 1:
            y_test = np.argmax(y_test, axis=-1)
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def _fitness_fn(self, rule_if: np.ndarray, rule_then: np.ndarray) -> np.ndarray:
        """Returns fitness for a given rule according to the RACER paper

        Args:
            rule_if (np.ndarray): If part of a rule (x)
            rule_then (np.ndarray): Then part of a rule (y)

        Returns:
            np.ndarray: Fitness score for the rule as defined in the RACER paper
        """
        n_covered, n_correct = self._confusion(rule_if, rule_then)
        accuracy = n_correct / n_covered
        coverage = n_covered / self._cardinality
        return self._alpha * accuracy + self._beta * coverage

    def _covered(self, X: np.ndarray, rule_if: np.ndarray) -> np.ndarray:
        """Returns indices of instances if `X` that are covered by `rule_if`.
        Note that rule covers instance if EITHER of the following holds in a bitwise manner:
        1. instance[i] == 0
        2. instance[i] == 1 AND rule[i] == 1

        Args:
            X (np.ndarray): Instances
            rule_if (np.ndarray): If part of rule (x)

        Returns:
            np.ndarray: An array containing indices in `X` that are covered by `rule_if`
        """
        covered = OR(NOT(X), AND(rule_if, X)).min(axis=-1)
        return covered

    def _confusion(
        self, rule_if: np.ndarray, rule_then: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns n_correct and n_covered for instances classified by a rule.

        Args:
            rule_if (np.ndarray): If part of rule (x)
            rule_then (np.ndarray): Then part of rule (y)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (n_covered, n_correct)
        """
        covered = self._covered(self._X, rule_if)
        n_covered = covered.sum()
        y_covered = self._y[covered]
        n_correct = XNOR(y_covered, rule_then).min(axis=-1).sum()
        return n_covered, n_correct

    def _get_majority(self) -> np.ndarray:
        """Return the majority rule_then from self._y

        Returns:
            np.ndarray: Majority rule_then
        """
        u, indices = np.unique(self._y, axis=0, return_inverse=True)
        return u[np.bincount(indices).argmax()]

    def _create_init_rules(self) -> None:
        """Creates an initial set of rules from theinput feature vectors"""
        self._extants_if = self._X.copy()
        self._extants_then = self._y.copy()
        self._extants_covered = np.zeros(len(self._X), dtype=bool)
        self._majority_then = self._get_majority()
        self._fitnesses = np.array(
            [
                self._fitness_fn(rule_if, rule_then)
                for rule_if, rule_then in zip(self._X, self._y)
            ]
        )

    def _composable(self, idx1: int, idx2: int) -> bool:
        """Returns true if two rules indicated by their indices are composable

        Args:
            idx1 (int): Index of the first rule
            idx2 (int): Index of the second rule

        Returns:
            bool: True if labels match and neither of the rules are covered. False otherwise.
        """
        labels_match = XNOR(self._extants_then[idx1], self._extants_then[idx2]).min()
        return (
            labels_match
            and not self._extants_covered[idx1]
            and not self._extants_covered[idx2]
        )

    def _process_rules(self, idx1: int, idx2: int) -> None:
        """Process two rules indiciated by their indices

        Args:
            idx1 (int): Index of the first rule
            idx2 (int): Index of the second rule
        """
        if self._composable(idx1, idx2):
            composition = self._compose(self._extants_if[idx1], self._extants_if[idx2])
            composition_fitness = self._fitness_fn(
                composition, self._extants_then[idx1]
            )
            if composition_fitness > np.maximum(
                self._fitnesses[idx1], self._fitnesses[idx2]
            ):
                self._update_extants(
                    idx1, composition, self._extants_then[idx1], composition_fitness
                )

    def _compose(self, rule1: np.ndarray, rule2: np.ndarray) -> np.ndarray:
        """Composes rule1 with rule2

        Args:
            rule1 (np.ndarray): The first rule
            rule2 (np.ndarray): The second rule

        Returns:
            np.ndarray: The composed rule which is simply the bitwise OR of the two rules
        """
        return OR(rule1, rule2)

    def _update_extants(
        self,
        index: int,
        new_rule_if: np.ndarray,
        new_rule_then: np.ndarray,
        new_rule_fitness: np.ndarray,
    ):
        """Remove all rules from current extants that are covered by `new_rule`.
        Then append new rule to extants.

        Args:
            index (int): Index of `new_rule`
            new_rule_if (np.ndarray): If part of `new_rule` (x)
            new_rule_then (np.ndarray): Then part of `new_rule` (y)
            new_rule_fitness (np.ndarray): Fitness of the `new_rule`
        """
        same_class_indices = self._class_indices[self._label_to_int(new_rule_then)]
        covered = self._covered(self._extants_if[same_class_indices], new_rule_if)
        self._extants_covered[same_class_indices[covered]] = True
        self._extants_covered[index] = False
        self._extants_if[index], self._extants_then[index], self._fitnesses[index] = (
            new_rule_if,
            new_rule_then,
            new_rule_fitness,
        )

    def _label_to_int(self, label: np.ndarray) -> int:
        """Converts dummy label to int

        Args:
            label (np.ndarray): Label to convert

        Returns:
            int: Converted label
        """
        return int(np.argmax(label))

    def _generalize_extants(self) -> None:
        """Generalize the extants by flipping every 0 to a 1 and checking if the fitness improves."""
        new_extants_if = np.zeros_like(self._extants_if, dtype=bool)
        for i in range(len(self._extants_if)):
            for j in range(len(self._extants_if[i])):
                if not self._extants_if[i][j]:
                    self._extants_if[i][j] = True
                    fitness = self._fitness_fn(
                        self._extants_if[i], self._extants_then[i]
                    )
                    if fitness > self._fitnesses[i]:
                        self._fitnesses[i] = fitness
                    else:
                        self._extants_if[i][j] = False
            new_extants_if[i] = self._extants_if[i]
        self._extants_if = new_extants_if

    def _finalize_rules(self) -> None:
        """Removes redundant rules to form the final ruleset"""
        temp_rules_if = self._final_rules_if
        temp_rules_then = self._final_rules_then
        temp_rules_fitnesses = self._fitnesses
        i = 0
        while i < len(temp_rules_if) - 1:
            mask = np.ones(len(temp_rules_if), dtype=bool)
            covered = self._covered(temp_rules_if[i + 1 :], temp_rules_if[i])
            mask[i + 1 :][covered] = False
            temp_rules_if, temp_rules_then, temp_rules_fitnesses = (
                temp_rules_if[mask],
                temp_rules_then[mask],
                temp_rules_fitnesses[mask],
            )
            i += 1

        self._final_rules_if, self._final_rules_then, self._fitnesses = (
            temp_rules_if,
            temp_rules_then,
            temp_rules_fitnesses,
        )
