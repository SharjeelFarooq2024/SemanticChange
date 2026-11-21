import argparse
import json
import zipfile
from pathlib import Path
import bcubed
import numpy as np
import tempfile
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from collections import Counter, defaultdict
from evaluation_utils import (
    load_subtask1_labels,
    calculate_JSD_distances,
    load_subtask2_labels,
)


def open_submission_zip(submission_path: Path, extraction_dir: Path, logs):
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission not found: {submission_path}")

    if not zipfile.is_zipfile(submission_path):
        raise ValueError(f"Submission is not a valid ZIP file: {submission_path}")

    with zipfile.ZipFile(submission_path, "r") as zip_file:
        files = [
            info.filename
            for info in zip_file.infolist()
            if not info.is_dir()
        ]

        logs.append({
            "level": "info",
            "message": f"Found {len(files)} files in submission.",
        })

        zip_file.extractall(extraction_dir)

    return files


def compute_spearman(gold_JSD, predicted_JSD, filename):
    predicted_scores = []
    gold_scores = []

    for item in gold_JSD:
        if item in predicted_JSD:
            predicted_scores.append(predicted_JSD[item])
            gold_scores.append(gold_JSD[item])
        else:
            word, time_period1, time_period2 = item.rsplit("_", 2)
            raise Exception(
                f"SUBTASK1b: Label missing in {filename} for word {word} "
                f"in time period {time_period1} or {time_period2}"
            )

    rho, pvalue = spearmanr(gold_scores, predicted_scores)

    return {
        "rho": float(rho) if not np.isnan(rho) else 0,
        "pvalue": float(pvalue),
    }


def compute_f1_score(gold_labels, predicted_labels, filename):
    # Group instances by word.
    by_word = defaultdict(list)

    for item in gold_labels:
        if item not in predicted_labels:
            word, sentence_id = item.split(";", 1)
            raise Exception(
                f"SUBTASK2: Label missing in {filename} "
                f"for {sentence_id} of word {word}"
            )

        word, sentence_id = item.split(";", 1)
        by_word[word].append(item)

    word_scores = {}

    for word, items in by_word.items():
        gold_word = [
            gold_labels[item]
            for item in items
        ]

        predicted_word = [
            predicted_labels[item]
            for item in items
        ]

        f1 = f1_score(
            gold_word,
            predicted_word,
            average='macro'
        )

        word_scores[word] = float(f1)

    # Macro-average F1 across words.
    macro_f1 = sum(word_scores.values()) / len(word_scores)

    return macro_f1


def compute_bcubed(gold_labels, predicted_labels, filename):
    # Group instances by word.
    by_word = defaultdict(list)

    for item in gold_labels:
        if item not in predicted_labels:
            word, sentence_id = item.split(";", 1)
            raise Exception(
                f"SUBTASK1a: Label missing in {filename} "
                f"for {sentence_id} of word {word}"
            )

        word, sentence_id = item.split(";", 1)
        by_word[word].append(item)

    word_scores = {}

    for word, items in by_word.items():
        gold_word = {}
        predicted_word = {}

        # bcubed expects:
        # entity -> set of cluster memberships
        for j, item in enumerate(items):
            gold_word[j] = set(gold_labels[item])
            predicted_word[j] = set(predicted_labels[item])

        precision = bcubed.precision(
            predicted_word,
            gold_word
        )
        recall = bcubed.recall(
            predicted_word,
            gold_word

        )
        fcubed = bcubed.fscore(
            precision,
            recall,
            beta=1.0,
        )

        word_scores[word] = {
            "precision": float(precision),
            "recall": float(recall),
            "fcubed": float(fcubed),
        }

    # Macro-average across words.
    precision = sum(
        scores["precision"]
        for scores in word_scores.values()
    ) / len(word_scores)

    recall = sum(
        scores["recall"]
        for scores in word_scores.values()
    ) / len(word_scores)

    fcubed = sum(
        scores["fcubed"]
        for scores in word_scores.values()
    ) / len(word_scores)

    return {
        "precision": precision,
        "recall": recall,
        "fcubed": fcubed,
    }

def filter_jsd_by_min_instances(jsd, gold_counts, min_instances=10):
    filtered = {}

    for item, value in jsd.items():
        word, period1, period2 = item.rsplit("_", 2)

        n1 = sum(gold_counts[word][period1].values())
        n2 = sum(gold_counts[word][period2].values())

        if n1 >= min_instances and n2 >= min_instances:
            filtered[item] = value

    return filtered


def evaluate_subtask1(gold_subtask1_path, predictions_subtask1_path):
    predicted_counts, gold_counts, predictions_subtask1, gold_subtask1 = load_subtask1_labels(predictions_subtask1_path, gold_subtask1_path)

    fcubed_subtask1a = compute_bcubed(
        gold_subtask1,
        predictions_subtask1,
        predictions_subtask1_path,
    )

    gold_JSD = calculate_JSD_distances(gold_counts)
    predicted_JSD = calculate_JSD_distances(predicted_counts)

    gold_JSD = filter_jsd_by_min_instances(
        gold_JSD,
        gold_counts,
        min_instances=10,
    )

    rho_subtask1b = compute_spearman(
        gold_JSD,
        predicted_JSD,
        predictions_subtask1_path,
    )["rho"]

    return fcubed_subtask1a, rho_subtask1b


def evaluate_subtask2(gold_subtask2_path, predictions_subtask2_path):
    predictions_subtask2, gold_subtask2 = load_subtask2_labels(predictions_subtask2_path,gold_subtask2_path)

    return compute_f1_score(
        gold_subtask2,
        predictions_subtask2,
        predictions_subtask2_path,
    )


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def add_averages(results):
    # Subtask 1a has three separate metrics.
    language_results_1a = [
        value
        for language, value in results["subtask1a"].items()
        if language != "average"
    ]

    if language_results_1a:
        results["subtask1a"]["average"] = {
            "precision": mean([
                x["precision"] for x in language_results_1a
            ]),
            "recall": mean([
                x["recall"] for x in language_results_1a
            ]),
            "fcubed": mean([
                x["fcubed"] for x in language_results_1a
            ]),
        }

    # Subtask 1b is a scalar per language.
    values_1b = [
        value
        for language, value in results["subtask1b"].items()
        if language != "average"
    ]

    if values_1b:
        results["subtask1b"]["average"] = mean(values_1b)

    # Subtask 2 is also a scalar per language.
    values_2 = [
        value
        for language, value in results["subtask2"].items()
        if language != "average"
    ]

    if values_2:
        results["subtask2"]["average"] = mean(values_2)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a submission."
    )

    parser.add_argument(
        "submission",
        type=Path,
        help="Submission ZIP file",
    )

    parser.add_argument(
        "gold_folder",
        type=Path,
        help="Folder containing gold data",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("evaluation_results.json"),
        help="Output JSON file",
    )

    args = parser.parse_args()

    allowed_languages = {"EN", "SV", "RU", "NL", "IT", "ES"}

    evaluation = {
        "results": {
            "subtask1a": {},
            "subtask1b": {},
            "subtask2": {},
        },
        "errors": [],
        "logs": [],
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        extraction_dir = Path(temp_dir)

        try:
            files = open_submission_zip(
                args.submission,
                extraction_dir,
                evaluation["logs"],
            )
        except Exception as exc:
            evaluation["errors"].append({
                "scope": "submission",
                "message": str(exc),
            })

            args.output.write_text(
                json.dumps(evaluation, indent=2),
                encoding="utf-8",
            )
            return

        subtask1_files = [
            f for f in files
            if f.endswith("_subtask1.jsonl")
        ]

        subtask2_files = [
            f for f in files
            if f.endswith("_subtask2.jsonl")
        ]

        languages_subtask1 = []

        for filename in subtask1_files:
            language = Path(filename).name.split("_")[0]

            if language not in allowed_languages:
                evaluation["logs"].append({
                    "level": "warning",
                    "message": (
                        f"Skipping {filename}: "
                        f"language {language} not supported."
                    ),
                })
            else:
                languages_subtask1.append(language)

        languages_subtask2 = []

        for filename in subtask2_files:
            language = Path(filename).name.split("_")[0]

            if language not in allowed_languages:
                evaluation["logs"].append({
                    "level": "warning",
                    "message": (
                        f"Skipping {filename}: "
                        f"language {language} not supported."
                    ),
                })
            else:
                languages_subtask2.append(language)

        # Avoid accidentally evaluating the same language twice.
        languages_subtask1 = sorted(set(languages_subtask1))
        languages_subtask2 = sorted(set(languages_subtask2))

        for language in languages_subtask1:
            gold_path = (
                args.gold_folder
                / language
                / "subtask1.jsonl"
            )

            predictions_path = (
                extraction_dir
                / f"{language}_subtask1.jsonl"
            )

            try:
                result_1a, result_1b = evaluate_subtask1(
                    gold_path,
                    predictions_path,
                )

                evaluation["results"]["subtask1a"][language] = result_1a
                evaluation["results"]["subtask1b"][language] = result_1b

                evaluation["logs"].append({
                    "level": "info",
                    "language": language,
                    "subtask": "subtask1",
                    "message": "Evaluation completed successfully.",
                })

            except Exception as exc:
                evaluation["errors"].append({
                    "language": language,
                    "subtask": "subtask1",
                    "message": str(exc),
                })

        for language in languages_subtask2:
            gold_path = (
                args.gold_folder
                / language
                / "subtask2.jsonl"
            )

            predictions_path = (
                extraction_dir
                / f"{language}_subtask2.jsonl"
            )

            try:
                result_2 = evaluate_subtask2(
                    gold_path,
                    predictions_path,
                )

                evaluation["results"]["subtask2"][language] = result_2

                evaluation["logs"].append({
                    "level": "info",
                    "language": language,
                    "subtask": "subtask2",
                    "message": "Evaluation completed successfully.",
                })

            except Exception as exc:
                evaluation["errors"].append({
                    "language": language,
                    "subtask": "subtask2",
                    "message": str(exc),
                })

        add_averages(evaluation["results"])

        args.output.write_text(
            json.dumps(
                evaluation,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        # Also emit the complete result to stdout.
        print(
            json.dumps(
                evaluation,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()