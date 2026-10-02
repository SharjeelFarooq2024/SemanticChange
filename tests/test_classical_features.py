import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.baselines.classical_features import build_feature_frame


def test_feature_frame_contains_eda_features():
    frame = build_feature_frame(pd.DataFrame({"text": ["En kvinna."], "word": ["kvinna"], "pos": ["NN"], "period_label": ["1880-1883"]}))
    assert {"token_count", "type_token_ratio", "stopword_ratio", "target_position"}.issubset(frame.columns)


def test_feature_frame_reports_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        build_feature_frame(pd.DataFrame({"text": ["text"]}))


def test_tfidf_vocabulary_is_fitted_on_training_data_only():
    pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True))])
    pipeline.fit(["common training token", "another common token"])
    assert "evaluationonly" not in pipeline.named_steps["tfidf"].vocabulary_


def test_tfidf_is_deterministic():
    first = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True))]).fit(["a common sentence", "another sentence"])
    second = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True))]).fit(["a common sentence", "another sentence"])
    assert first.named_steps["tfidf"].vocabulary_ == second.named_steps["tfidf"].vocabulary_


def test_ablated_preprocessor_excludes_word_column():
    from src.baselines.classical_features import build_ablated_preprocessor, build_interpretable_preprocessor
    config = {
        "classical": {
            "word_ngram_range": [1, 2],
            "min_df": 1,
            "max_features": 1000,
        }
    }
    df = build_feature_frame(pd.DataFrame({
        "text": ["En kvinna stod där."],
        "word": ["kvinna"],
        "pos": ["NN"],
        "period_label": ["1880-1883"]
    }))
    pre_interp = build_interpretable_preprocessor(config).fit(df)
    pre_ablated = build_ablated_preprocessor(config).fit(df)
    interp_names = pre_interp.get_feature_names_out()
    ablated_names = pre_ablated.get_feature_names_out()
    assert any("categorical__word_" in name for name in interp_names)
    assert not any("categorical__word_" in name for name in ablated_names)