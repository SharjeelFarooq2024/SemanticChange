import re
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .classical_features import (
    build_ablated_preprocessor,
    build_feature_frame,
    build_interpretable_preprocessor,
    feature_names,
    get_categorical_features,
    get_numeric_features,
)
from .common import join_subtask2_labels, load_baseline_bundle, load_config, prepare_output_dirs, set_seed, write_json


def plot_top_coefficients(
    coef_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
    top_n: int = 20,
) -> str:
    """Save a horizontal bar chart of the top-N positive and negative LR coefficients.

    This connects EDA lexical observations to model decisions:
    positive coefficients correspond to tokens most predictive of label=1
    (usage fits the modern/target sense), negative coefficients predict label=0.
    """
    model_coefs = coef_df[coef_df["model"] == model_name].copy()
    if model_coefs.empty:
        return ""
    top_pos = model_coefs.nlargest(top_n, "coefficient")
    top_neg = model_coefs.nsmallest(top_n, "coefficient")
    combined = pd.concat([top_neg, top_pos]).drop_duplicates("feature")
    combined = combined.sort_values("coefficient")

    fig, ax = plt.subplots(figsize=(10, max(6, len(combined) * 0.32)))
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in combined["coefficient"]]
    ax.barh(combined["feature"], combined["coefficient"], color=colors, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Logistic Regression coefficient", fontsize=11)
    ax.set_title(
        f"{model_name}\nTop-{top_n} positive (green=label 1) and negative (red=label 0) features",
        fontsize=11,
    )
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_feature_importances(
    imp_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
    top_n: int = 20,
) -> str:
    """Save a horizontal bar chart of top-N tree-based feature importances.

    Connects tree-based decision criteria to EDA observations:
    ranks which features provide the largest reduction in Gini impurity.
    """
    model_imps = imp_df[imp_df["model"] == model_name].copy()
    if model_imps.empty:
        return ""
    top = model_imps.nlargest(top_n, "importance").sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.32)))
    ax.barh(top["feature"], top["importance"], color="#2980b9", edgecolor="none")
    ax.set_xlabel("Gini Feature Importance (mean decrease in impurity)", fontsize=10)
    ax.set_title(
        f"{model_name}\nTop-{top_n} features by Random Forest importance",
        fontsize=11,
    )
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _metrics(y_true, y_pred) -> dict:
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision,
        "recall": recall,
        "precision_macro": precision,
        "recall_macro": recall,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def _split(df: pd.DataFrame, config: dict):
    stratify = df["label_value"] if config["evaluation"].get("stratify", True) else None
    return train_test_split(df, test_size=config["evaluation"]["test_size"], random_state=config["evaluation"]["random_state"], stratify=stratify)



def run(project_root: Path | None = None) -> dict:
    root = project_root or Path(__file__).resolve().parents[2]
    config = load_config(root)
    set_seed(int(config["seed"]))
    prepare_output_dirs(root, config)
    bundle = load_baseline_bundle(root, config)
    data, diagnostics = join_subtask2_labels(bundle["usage"], bundle["subtask2"])
    if data["label_value"].nunique() < 2:
        raise ValueError("Subtask 2 needs both binary classes after joining labels")
    train, test = _split(data, config)
    output = root / config["outputs"]["classical"]
    runs = []
    predictions = []

    text_models = {
        "majority": (DummyClassifier(strategy="most_frequent"), "majority"),
        "tfidf_logistic_regression": (Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=tuple(config["classical"]["word_ngram_range"]), lowercase=True, min_df=config["classical"]["min_df"], max_features=config["classical"]["max_features"], sublinear_tf=True)),
            ("model", LogisticRegression(max_iter=config["classical"]["max_iter"], class_weight=config["classical"]["class_weight"], random_state=config["seed"])),
        ]), "word_tfidf"),
        "word_char_tfidf_linear_svm": (Pipeline([
            ("tfidf", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=tuple(config["classical"]["word_ngram_range"]), lowercase=True, min_df=config["classical"]["min_df"], max_features=config["classical"]["max_features"], sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char", ngram_range=tuple(config["classical"]["char_ngram_range"]), lowercase=True, min_df=config["classical"]["min_df"], max_features=config["classical"]["max_features"], sublinear_tf=True)),
            ])),
            ("model", LinearSVC(class_weight=config["classical"]["class_weight"], random_state=config["seed"])),
        ]), "word_and_character_tfidf"),
        "tfidf_complement_nb": (Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=tuple(config["classical"]["word_ngram_range"]), lowercase=True, min_df=config["classical"]["min_df"], max_features=config["classical"]["max_features"], sublinear_tf=True)),
            ("model", ComplementNB(norm=True)),
        ]), "word_tfidf"),
    }
    for name, (model, representation) in text_models.items():
        model.fit(train["text"], train["label_value"])
        prediction = model.predict(test["text"])
        frame = test[["sentence_id", "word", "period_label", "label_value"]].copy()
        frame["model"] = name
        frame["prediction"] = prediction
        predictions.append(frame)
        runs.append({"model": name, "representation": representation, "train_size": len(train), "test_size": len(test), **_metrics(test["label_value"], prediction)})

    # Full EDA feature models (TF-IDF + numeric EDA features + categorical word/pos/period)
    feature_train = build_feature_frame(train, config)
    feature_test = build_feature_frame(test, config)

    interpretable_lr = Pipeline([
        ("features", build_interpretable_preprocessor(config)),
        ("model", LogisticRegression(max_iter=config["classical"]["max_iter"], class_weight=config["classical"]["class_weight"], random_state=config["seed"])),
    ])
    interpretable_lr.fit(feature_train, train["label_value"])
    lr_pred = interpretable_lr.predict(feature_test)
    frame_lr = test[["sentence_id", "word", "period_label", "label_value"]].copy()
    frame_lr["model"] = "eda_features_logistic_regression"
    frame_lr["prediction"] = lr_pred
    predictions.append(frame_lr)
    runs.append({"model": "eda_features_logistic_regression", "representation": "tfidf_plus_eda_features", "train_size": len(train), "test_size": len(test), **_metrics(test["label_value"], lr_pred)})

    rf_n_est = int(config["classical"].get("random_forest_n_estimators", 100))
    rf_max_d = int(config["classical"].get("random_forest_max_depth", 15))
    interpretable_rf = Pipeline([
        ("features", build_interpretable_preprocessor(config)),
        ("model", RandomForestClassifier(n_estimators=rf_n_est, max_depth=rf_max_d, class_weight=config["classical"]["class_weight"], random_state=config["seed"], n_jobs=-1)),
    ])
    interpretable_rf.fit(feature_train, train["label_value"])
    rf_pred = interpretable_rf.predict(feature_test)
    frame_rf = test[["sentence_id", "word", "period_label", "label_value"]].copy()
    frame_rf["model"] = "eda_features_random_forest"
    frame_rf["prediction"] = rf_pred
    predictions.append(frame_rf)
    runs.append({"model": "eda_features_random_forest", "representation": "tfidf_plus_eda_features_rf", "train_size": len(train), "test_size": len(test), **_metrics(test["label_value"], rf_pred)})

    # -------------------------------------------------------------------------
    # Word-identity ablation: mask ALL surface forms of the target word
    # -------------------------------------------------------------------------
    # Original bug: the simple \b{word}\b regex only masks the exact form stored
    # in the 'word' column.  For a Swedish lemma like "driva", inflected forms
    # (driver, drev, drivit, driven, drivande, ...) survived into the TF-IDF
    # matrix and let the model recover word identity from context — defeating
    # the purpose of the ablation.
    #
    # Fix: _morphological_forms() generates a compact set of common Swedish
    # surface variants (definite/indefinite endings, plural, genitive, past
    # participle stems, present participle) for both nouns and verbs.  The
    # expander is language-agnostic in principle — it applies suffix rules
    # rather than a lexicon — and is safe to run without any extra dependency.
    # Every generated form is masked with the special token [TARGET] before
    # TF-IDF vectorisation, ensuring the model cannot reconstruct word identity
    # from any morphological variant in the context window.

    @lru_cache(maxsize=None)
    def _morphological_forms(word: str) -> frozenset[str]:
        """Return a set of likely Swedish surface forms for *word*.

        Covers the most productive morphological patterns for Swedish nouns,
        adjectives, and verbs so that no inflected form leaks word identity
        into the ablated TF-IDF matrix.
        """
        w = word.lower().rstrip()
        forms: set[str] = {w}  # base form always included

        # --- Noun suffixes (strong & weak declensions) --------------------
        # Definite singular: + n/en/et
        forms.update({w + "n", w + "en", w + "et"})
        # Definite plural / plural indefinite: + ar/er/or/r + (na/na/na)
        for pl in ("ar", "er", "or", "r"):
            forms.add(w + pl)
            forms.add(w + pl + "na")
        # Genitive (add s to base and each generated form — productive rule)
        forms.update({f + "s" for f in set(forms)})

        # --- Verb suffixes -----------------------------------------------
        # Present: +er / +ar / +r
        forms.update({w + "er", w + "ar", w + "r"})
        # Past weak: +ade / +de / +te
        forms.update({w + "ade", w + "de", w + "te"})
        # Supine / past participle: +at / +t / +tt / +it
        forms.update({w + "at", w + "t", w + "tt", w + "it"})
        # Present participle: +ande / +ende
        forms.update({w + "ande", w + "ende"})
        # Passive: +as / +es
        forms.update({w + "as", w + "es"})

        # --- Stem-final e-drop (common in Swedish strong verbs) -----------
        # e.g. "skriva" → stem "skriv" → "skriver", "skrev" (handled above
        # for the already-stemmed form; here we also try dropping final 'a')
        if w.endswith("a") and len(w) > 2:
            stem = w[:-1]
            forms.update({
                stem, stem + "er", stem + "ar", stem + "r",
                stem + "s", stem + "de", stem + "te",
            })

        return frozenset(forms)

    def _build_mask_pattern(word: str) -> re.Pattern:
        """Compile a single alternation regex for all morphological forms."""
        all_forms = _morphological_forms(word.lower())
        # Sort longest-first so longer forms match before shorter prefixes
        alternation = "|".join(re.escape(f) for f in sorted(all_forms, key=len, reverse=True))
        return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)

    def _mask_word_morphological(row) -> str:
        """Replace every morphological variant of the target word with [TARGET]."""
        w = str(row.get("word", "") or "").strip()
        t = str(row.get("text", "") or "")
        if not w:
            return t
        pattern = _build_mask_pattern(w)
        return pattern.sub("[TARGET]", t)

    train_ablated = train.copy()
    train_ablated["text"] = train_ablated.apply(_mask_word_morphological, axis=1)
    test_ablated = test.copy()
    test_ablated["text"] = test_ablated.apply(_mask_word_morphological, axis=1)
    feature_train_ablated = build_feature_frame(train_ablated, config)
    feature_test_ablated = build_feature_frame(test_ablated, config)

    ablated_lr = Pipeline([
        ("features", build_ablated_preprocessor(config)),
        ("model", LogisticRegression(max_iter=config["classical"]["max_iter"], class_weight=config["classical"]["class_weight"], random_state=config["seed"])),
    ])
    ablated_lr.fit(feature_train_ablated, train["label_value"])
    ablated_pred = ablated_lr.predict(feature_test_ablated)
    frame_ablated = test[["sentence_id", "word", "period_label", "label_value"]].copy()
    frame_ablated["model"] = "ablated_no_word_id_logistic_regression"
    frame_ablated["prediction"] = ablated_pred
    predictions.append(frame_ablated)
    runs.append({"model": "ablated_no_word_id_logistic_regression", "representation": "context_and_eda_without_word_identity", "train_size": len(train), "test_size": len(test), **_metrics(test["label_value"], ablated_pred)})

    # Linear model coefficients (Logistic Regression)
    coefficient_frames = []
    for model_name, model in (("tfidf_logistic_regression", text_models["tfidf_logistic_regression"][0]), ("eda_features_logistic_regression", interpretable_lr)):
        fitted = model.named_steps["model"]
        names = model.named_steps["tfidf"].get_feature_names_out() if model_name == "tfidf_logistic_regression" else feature_names(model.named_steps["features"])
        coefficients = pd.DataFrame({"model": model_name, "feature": names, "coefficient": fitted.coef_[0]})
        coefficients["direction"] = coefficients["coefficient"].map(lambda value: "positive" if value >= 0 else "negative")
        coefficient_frames.append(pd.concat([coefficients.nlargest(25, "coefficient"), coefficients.nsmallest(25, "coefficient")]))
    coefficient_output = pd.concat(coefficient_frames, ignore_index=True).drop_duplicates(["model", "feature"])
    coefficient_output.to_csv(output / "subtask2_coefficients.csv", index=False)

    # Tree-based Gini feature importance (Random Forest) - keep top 100 features for interpretability & EDA link
    rf_feature_names = feature_names(interpretable_rf.named_steps["features"])
    rf_importances = pd.DataFrame({
        "model": "eda_features_random_forest",
        "feature": rf_feature_names,
        "importance": interpretable_rf.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False)
    rf_importances.head(100).to_csv(output / "subtask2_rf_feature_importances.csv", index=False)

    pd.DataFrame(runs).to_csv(output / "subtask2_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "subtask2_predictions.csv", index=False)

    # Save feature visualizations
    plot_top_coefficients(coefficient_output, "tfidf_logistic_regression", output / "subtask2_tfidf_lr_coefficients.png", top_n=20)
    plot_top_coefficients(coefficient_output, "eda_features_logistic_regression", output / "subtask2_eda_lr_coefficients.png", top_n=20)
    plot_feature_importances(rf_importances, "eda_features_random_forest", output / "subtask2_rf_feature_importance.png", top_n=20)

    all_models = list(text_models) + ["eda_features_logistic_regression", "eda_features_random_forest", "ablated_no_word_id_logistic_regression"]
    write_json(output / "subtask2_metadata.json", {
        "seed": config["seed"],
        "split": config["evaluation"],
        "split_sizes": {"train": len(train), "test": len(test)},
        "stratified_label_counts": {"train": train["label_value"].value_counts().to_dict(), "test": test["label_value"].value_counts().to_dict()},
        "models": all_models,
        "coefficient_count": len(coefficient_output),
        "rf_importance_count": min(len(rf_importances), 100),
        "eda_numeric_features": get_numeric_features(config),
        "eda_categorical_features": get_categorical_features(config),
        "data_diagnostics": diagnostics,
    })
    return {"metrics": runs, "diagnostics": diagnostics}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))