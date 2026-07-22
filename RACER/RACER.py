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
        fallback="majority",
    ):
        """Initialize the RACER class

        Args:
            alpha (float, optional): Value of alpha according to the RACER paper. Defaults to 0.9.
            suppress_warnings (bool, optional): Whether to suppress any warnings raised during prediction. Defaults to False.
            benchmark (bool, optional): Whether to time the `fit` method for benchmark purposes. Defaults to False.
            fallback (str, optional): How to label instances not covered by a rule.
                ``"majority"`` preserves the original behavior; ``"partial-match"``
                selects by overlap, fitness, then rule order. Defaults to ``"majority"``.
        """
        assert fallback in [
            "majority",
            "partial-match",
        ], "`fallback` must either be 'majority' or 'partial-match'."
        self._alpha, self._beta = alpha, 1.0 - alpha
        self._suppress_warnings = suppress_warnings
        self._benchmark = benchmark
        self._fallback = fallback
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
        use_packed = bool(getattr(self, "_use_packed_coverage", True))
        self._coverage_backend = "uint64" if use_packed else "boolean-reference"
        self._X_packed = self._pack_bits(self._X) if use_packed else None

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
        if self._extants_if_packed is not None:
            self._extants_if_packed = self._extants_if_packed[independent_indices]

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
        self._final_rules_if_packed = (
            self._extants_if_packed[args]
            if self._extants_if_packed is not None
            else None
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
        all_found = found.sum() == len(X)
        X_packed = (
            self._pack_bits(X)
            if getattr(self, "_final_rules_if_packed", None) is not None
            else None
        )
        for i in range(len(self._final_rules_if)):
            covered = (
                self._covered_packed(X_packed, self._final_rules_if_packed[i])
                if X_packed is not None
                else self._covered(X, self._final_rules_if[i])
            )
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
                if self._fallback == "partial-match" and len(self._final_rules_if):
                    print(
                        "These instances will be labelled using the best partial-matching rule."
                    )
                else:
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
        """Find the configured fallback label for an uncovered instance.

        Args:
            X (np.ndarray): Input rule `X`

        Returns:
            np.ndarray: Matched rule
        """
        if self._fallback == "majority" or not len(self._final_rules_if):
            return self._majority_then

        denominator = X.sum()
        intersections = AND(self._final_rules_if, X).sum(axis=-1)
        overlaps = (
            intersections / denominator
            if denominator
            else np.zeros(len(self._final_rules_if))
        )
        order = np.lexsort(
            (
                np.arange(len(self._final_rules_if)),
                -self._fitnesses,
                -overlaps,
            )
        )
        return self._final_rules_then[order[0]]

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

    def _fitness_fn(
        self, rule_if: np.ndarray, rule_then: np.ndarray, packed_rule=None
    ) -> np.ndarray:
        """Returns fitness for a given rule according to the RACER paper

        Args:
            rule_if (np.ndarray): If part of a rule (x)
            rule_then (np.ndarray): Then part of a rule (y)

        Returns:
            np.ndarray: Fitness score for the rule as defined in the RACER paper
        """
        n_covered, n_correct = self._confusion(
            rule_if, rule_then, packed_rule=packed_rule
        )
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

    def _pack_bits(self, bits: np.ndarray) -> np.ndarray:
        """Pack the final feature axis into zero-padded uint64 words."""
        bits = np.asarray(bits, dtype=bool)
        if bits.ndim not in (1, 2):
            raise ValueError("packed RACER bits must be one- or two-dimensional")
        words = (bits.shape[-1] + 63) // 64
        packed_bytes = np.packbits(bits, axis=-1, bitorder="little")
        byte_width = words * np.dtype(np.uint64).itemsize
        if packed_bytes.shape[-1] < byte_width:
            padding = [(0, 0)] * packed_bytes.ndim
            padding[-1] = (0, byte_width - packed_bytes.shape[-1])
            packed_bytes = np.pad(packed_bytes, padding, mode="constant")
        contiguous = np.ascontiguousarray(packed_bytes)
        return contiguous.view(np.uint64).reshape(*bits.shape[:-1], words)

    def _covered_packed(self, X: np.ndarray, rule_if: np.ndarray) -> np.ndarray:
        """Return coverage via ``(instance AND NOT rule) == 0`` per word."""
        uncovered = AND(X, NOT(rule_if))
        return np.equal(uncovered, 0).all(axis=-1)

    def _training_covered(
        self, rule_if: np.ndarray, packed_rule=None
    ) -> np.ndarray:
        if self._X_packed is not None:
            if packed_rule is None:
                packed_rule = self._pack_bits(rule_if)
            return self._covered_packed(self._X_packed, packed_rule)
        return self._covered(self._X, rule_if)

    def _confusion(
        self, rule_if: np.ndarray, rule_then: np.ndarray, packed_rule=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns n_covered and n_correct for instances classified by a rule.

        Args:
            rule_if (np.ndarray): If part of rule (x)
            rule_then (np.ndarray): Then part of rule (y)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (n_covered, n_correct)
        """
        covered = self._training_covered(rule_if, packed_rule=packed_rule)
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
        self._extants_if_packed = (
            self._pack_bits(self._extants_if)
            if self._X_packed is not None
            else None
        )
        self._extants_then = self._y.copy()
        self._extants_covered = np.zeros(len(self._X), dtype=bool)
        self._majority_then = self._get_majority()
        self._fitnesses = np.array(
            [
                self._fitness_fn(
                    rule_if,
                    rule_then,
                    packed_rule=(
                        self._extants_if_packed[index]
                        if self._extants_if_packed is not None
                        else None
                    ),
                )
                for index, (rule_if, rule_then) in enumerate(zip(self._X, self._y))
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
            composition_packed = (
                OR(
                    self._extants_if_packed[idx1],
                    self._extants_if_packed[idx2],
                )
                if self._extants_if_packed is not None
                else None
            )
            composition_fitness = self._fitness_fn(
                composition,
                self._extants_then[idx1],
                packed_rule=composition_packed,
            )
            if composition_fitness > np.maximum(
                self._fitnesses[idx1], self._fitnesses[idx2]
            ):
                self._update_extants(
                    idx1,
                    composition,
                    self._extants_then[idx1],
                    composition_fitness,
                    packed_rule=composition_packed,
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
        packed_rule=None,
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
        if self._extants_if_packed is not None:
            if packed_rule is None:
                packed_rule = self._pack_bits(new_rule_if)
            covered = self._covered_packed(
                self._extants_if_packed[same_class_indices], packed_rule
            )
        else:
            covered = self._covered(
                self._extants_if[same_class_indices], new_rule_if
            )
        self._extants_covered[same_class_indices[covered]] = True
        self._extants_covered[index] = False
        self._extants_if[index], self._extants_then[index], self._fitnesses[index] = (
            new_rule_if,
            new_rule_then,
            new_rule_fitness,
        )
        if self._extants_if_packed is not None:
            self._extants_if_packed[index] = packed_rule

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
                    if self._extants_if_packed is not None:
                        word_index, word_bit = divmod(j, 64)
                        bit = np.uint64(1) << np.uint64(word_bit)
                        self._extants_if_packed[i, word_index] |= bit
                    fitness = self._fitness_fn(
                        self._extants_if[i],
                        self._extants_then[i],
                        packed_rule=(
                            self._extants_if_packed[i]
                            if self._extants_if_packed is not None
                            else None
                        ),
                    )
                    if fitness > self._fitnesses[i]:
                        self._fitnesses[i] = fitness
                    else:
                        self._extants_if[i][j] = False
                        if self._extants_if_packed is not None:
                            self._extants_if_packed[i, word_index] &= ~bit
            new_extants_if[i] = self._extants_if[i]
        self._extants_if = new_extants_if

    def _finalize_rules(self) -> None:
        """Removes redundant rules to form the final ruleset"""
        temp_rules_if = self._final_rules_if
        temp_rules_if_packed = self._final_rules_if_packed
        temp_rules_then = self._final_rules_then
        temp_rules_fitnesses = self._fitnesses
        i = 0
        while i < len(temp_rules_if) - 1:
            mask = np.ones(len(temp_rules_if), dtype=bool)
            covered = (
                self._covered_packed(
                    temp_rules_if_packed[i + 1 :], temp_rules_if_packed[i]
                )
                if temp_rules_if_packed is not None
                else self._covered(temp_rules_if[i + 1 :], temp_rules_if[i])
            )
            mask[i + 1 :][covered] = False
            temp_rules_if, temp_rules_then, temp_rules_fitnesses = (
                temp_rules_if[mask],
                temp_rules_then[mask],
                temp_rules_fitnesses[mask],
            )
            if temp_rules_if_packed is not None:
                temp_rules_if_packed = temp_rules_if_packed[mask]
            i += 1

        self._final_rules_if, self._final_rules_then, self._fitnesses = (
            temp_rules_if,
            temp_rules_then,
            temp_rules_fitnesses,
        )
        self._final_rules_if_packed = temp_rules_if_packed
