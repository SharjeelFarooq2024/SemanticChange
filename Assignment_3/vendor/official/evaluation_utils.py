import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Hashable
from scipy.spatial.distance import jensenshannon


def period_sort_key(period_label: str):
    """Sort period labels by their first four-digit year."""
    match = re.search(r"\d{4}", period_label)

    if match is None:
        return math.inf, period_label

    return int(match.group()), period_label



def normalize_distribution(counts: Counter, labels: list[Hashable]):
    """Convert fractional label counts into probabilities."""
    total = sum(counts.values())

    if total <= 0:
        raise ValueError("Cannot normalize an empty distribution")

    return [counts.get(label, 0.0) / total for label in labels]



def calculate_JSD_distances(counts: dict[str, dict[str, Counter]]):

    results = {}

    for word in sorted(counts):
        period_counts = counts[word]
        ordered_periods = sorted(period_counts, key=period_sort_key)

        if len(ordered_periods) < 2:
            raise Exception(f'{word} has less than 2 time periods.')

        for previous_period, current_period in zip(ordered_periods,ordered_periods[1:]):
            previous_counts = period_counts[previous_period]
            current_counts = period_counts[current_period]

            # Use the union of labels found in the two periods.
            labels = sorted(
                set(previous_counts) | set(current_counts),
                key=lambda value: (str(type(value)), str(value)),
            )

            previous_distribution = normalize_distribution(
                previous_counts,
                labels,
            )
            current_distribution = normalize_distribution(
                current_counts,
                labels,
            )

            distance = jensenshannon(
                previous_distribution,
                current_distribution,
                base=2.0,
            )

            results[f'{word}_{previous_period}_{current_period}'] = distance

    return results

def load_subtask2_labels(
    input_path: Path,
    gold_path: Path):
    """
    Load gold and predicted Task 2 labels.

    Gold determines the expected (word, sentence_id) usages.

    Returns:
        predicted_labels:
            Predicted labels indexed by "word;sentence_id".

        gold_labels:
            Gold labels indexed by "word;sentence_id".
    """

    def load_file(path: Path) -> dict[tuple[str, str], int]:
        labels_by_usage = {}

        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    observation = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {path}: {error}"
                    ) from error

                required_fields = {"word", "sentence_id", "label"}
                missing_fields = required_fields - observation.keys()

                if missing_fields:
                    raise ValueError(
                        f"Missing field(s) on line {line_number} of {path}: "
                        f"{', '.join(sorted(missing_fields))}"
                    )

                word = observation["word"]
                sentence_id = observation["sentence_id"]
                label = observation["label"]

                if not isinstance(word, str) or not word:
                    raise ValueError(
                        f"'word' must be a non-empty string "
                        f"on line {line_number} of {path}"
                    )

                if not isinstance(sentence_id, str) or not sentence_id:
                    raise ValueError(
                        f"'sentence_id' must be a non-empty string "
                        f"on line {line_number} of {path}"
                    )

                # Explicit type check is necessary because
                # isinstance(True, int) is True.
                if (
                    not isinstance(label, int)
                    or isinstance(label, bool)
                    or label not in (0, 1)
                ):
                    raise ValueError(
                        f"'label' must be integer 0 or 1 "
                        f"on line {line_number} of {path}"
                    )

                key = (word, sentence_id)

                # Reject duplicates even when the labels are identical.
                if key in labels_by_usage:
                    raise ValueError(
                        f"Duplicate usage {key!r} "
                        f"on line {line_number} of {path}"
                    )

                labels_by_usage[key] = label

        return labels_by_usage

    gold = load_file(gold_path)
    predicted = load_file(input_path)

    expected_keys = set(gold)
    predicted_keys = set(predicted)

    # Reject prediction usages that don't exist in gold.
    unknown_keys = predicted_keys - expected_keys

    if unknown_keys:
        word, sentence_id = sorted(unknown_keys)[0]
        raise ValueError(
            f"Unknown usage in predictions: "
            f"word={word!r}, sentence_id={sentence_id!r}"
        )

    # Reject usages expected by gold but missing from predictions.
    missing_keys = expected_keys - predicted_keys

    if missing_keys:
        word, sentence_id = sorted(missing_keys)[0]
        raise ValueError(
            f"Missing expected usage in predictions: "
            f"word={word!r}, sentence_id={sentence_id!r}"
        )

    predicted_labels = {
        f"{word};{sentence_id}": predicted[(word, sentence_id)]
        for word, sentence_id in gold
    }

    gold_labels = {
        f"{word};{sentence_id}": gold[(word, sentence_id)]
        for word, sentence_id in gold
    }

    return predicted_labels, gold_labels


def load_subtask1_labels(
    input_path: Path,
    gold_path: Path,
):
    """
    Load gold and predicted Task 1 labels.

    Gold determines the expected usages and their periods.

    Returns:
        predicted_counts:
            Fractional predicted label counts by word and period.

        gold_counts:
            Fractional gold label counts by word and period.

        predicted_labels:
            Predicted labels indexed by "word;sentence_id".

        gold_labels:
            Gold labels indexed by "word;sentence_id".
    """

    def load_file(
        path: Path,
        *,
        require_period: bool,
    ):
        observations = {}

        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    observation = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {path}: {error}"
                    ) from error

                required_fields = {"word", "sentence_id", "label"}

                if require_period:
                    required_fields.add("period_label")

                missing_fields = required_fields - observation.keys()

                if missing_fields:
                    raise ValueError(
                        f"Missing field(s) on line {line_number} of {path}: "
                        f"{', '.join(sorted(missing_fields))}"
                    )

                word = observation["word"]
                sentence_id = observation["sentence_id"]
                labels = observation["label"]

                if not isinstance(word, str) or not word:
                    raise ValueError(
                        f"'word' must be a non-empty string "
                        f"on line {line_number} of {path}"
                    )

                if not isinstance(sentence_id, str) or not sentence_id:
                    raise ValueError(
                        f"'sentence_id' must be a non-empty string "
                        f"on line {line_number} of {path}"
                    )

                if not isinstance(labels, list) or not labels:
                    raise ValueError(
                        f"'label' must be a non-empty list "
                        f"on line {line_number} of {path}"
                    )

                # bool must be checked explicitly because
                # isinstance(True, int) is True.
                if any(
                    not isinstance(label, int) or isinstance(label, bool)
                    for label in labels
                ):
                    raise ValueError(
                        f"All labels must be integers (not booleans) "
                        f"on line {line_number} of {path}"
                    )

                if len(labels) != len(set(labels)):
                    raise ValueError(
                        f"Labels must be distinct "
                        f"on line {line_number} of {path}"
                    )

                if require_period:
                    period = observation["period_label"]

                    if not isinstance(period, str) or not period:
                        raise ValueError(
                            f"'period_label' must be a non-empty string "
                            f"on line {line_number} of {path}"
                        )

                key = (word, sentence_id)

                if key in observations:
                    raise ValueError(
                        f"Duplicate usage {key!r} "
                        f"on line {line_number} of {path}"
                    )

                observations[key] = observation

        return observations

    # Gold establishes the expected usages and periods.
    gold = load_file(gold_path, require_period=True)
    predicted = load_file(input_path, require_period=False)

    expected_keys = set(gold)
    predicted_keys = set(predicted)

    unknown_keys = predicted_keys - expected_keys

    if unknown_keys:
        word, sentence_id = sorted(unknown_keys)[0]
        raise ValueError(
            f"Unknown usage in predictions: "
            f"word={word!r}, sentence_id={sentence_id!r}"
        )

    missing_keys = expected_keys - predicted_keys

    if missing_keys:
        word, sentence_id = sorted(missing_keys)[0]
        raise ValueError(
            f"Missing expected usage in predictions: "
            f"word={word!r}, sentence_id={sentence_id!r}"
        )

    predicted_counts: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    gold_counts: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    predicted_labels: dict[str, list[int]] = {}
    gold_labels: dict[str, list[int]] = {}

    for (word, sentence_id), gold_observation in gold.items():
        period = gold_observation["period_label"]

        predicted_label = predicted[(word, sentence_id)]["label"]
        gold_label = gold_observation["label"]

        usage_key = f"{word};{sentence_id}"

        predicted_labels[usage_key] = predicted_label
        gold_labels[usage_key] = gold_label

        # Predicted fractional counts.
        predicted_weight = 1.0 / len(predicted_label)

        for label in predicted_label:
            predicted_counts[word][period][label] += predicted_weight

        # Gold fractional counts.
        gold_weight = 1.0 / len(gold_label)

        for label in gold_label:
            gold_counts[word][period][label] += gold_weight

    predicted_counts = {
        word: dict(period_counts)
        for word, period_counts in predicted_counts.items()
    }

    gold_counts = {
        word: dict(period_counts)
        for word, period_counts in gold_counts.items()
    }

    return (
        predicted_counts,
        gold_counts,
        predicted_labels,
        gold_labels,
    )