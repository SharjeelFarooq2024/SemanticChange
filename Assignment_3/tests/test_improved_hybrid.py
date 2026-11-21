"""Checks for leakage, calibration application and feature-block independence."""
import json

import numpy as np
import pandas as pd
from scipy import sparse

from Assignment_3.src.improved_hybrid import (HybridFeatureBuilder, apply_thresholds,
    combine_blocks, fit_thresholds, query_document_features)
from Assignment_3.src.runtime import ROOT, load_data


def config():
    return json.loads((ROOT / "Assignment_3/configs/advanced_hybrid.json").read_text(encoding="utf-8"))


def test_features_ignore_gold_labels():
    data, _ = load_data()
    frame = data.groupby("word", sort=False).head(2).copy()
    first = query_document_features(frame, config()["reference_run"])
    frame["label_value"] = 1-frame.label_value
    second = query_document_features(frame, config()["reference_run"])
    for a,b in zip(first, second):
        np.testing.assert_array_equal(a,b)


def test_inner_vocabulary_never_fits_validation_tokens():
    data, _ = load_data()
    fit = data.head(8).copy()
    val = data.iloc[8:10].copy()
    # Keep identity and original embeddings; only the vectorizer sees this view.
    builder = HybridFeatureBuilder(config())
    builder.fit_transform(fit)
    assert "validationuniquetoken" not in builder.word.vocabulary_
    before = builder.word.vocabulary_.copy()
    builder.word.transform(["validationuniquetoken"])
    assert builder.word.vocabulary_ == before


def test_feature_ablation_drops_retrieval_and_embedding_blocks():
    blocks = {"context": sparse.csr_matrix([[1.,2.]]), "retrieval": sparse.csr_matrix([[3.]]),
              "embedding": sparse.csr_matrix([[4.,5.]])}
    context = combine_blocks(blocks, {"context_weight": 1., "retrieval_weight": 0., "embedding_weight": 0.})
    assert context.shape == (1,2)
    np.testing.assert_array_equal(context.toarray(), [[1.,2.]])
    hybrid = combine_blocks(blocks, {"context_weight": 1., "retrieval_weight": .25, "embedding_weight": .5})
    np.testing.assert_array_equal(hybrid.toarray(), [[1.,2.,.75,2.,2.5]])


def test_unknown_lemma_uses_global_threshold_and_calibration_does_not_need_test_gold():
    calibration = {"global": .2, "per_word": {"known": .8}}
    np.testing.assert_array_equal(apply_thresholds(["known", "new"], [.5,.5], calibration), [0,1])
    learned = fit_thresholds(np.array(["a"]*4), [0,0,1,1], [-2.,-1.,1.,2.], .5)
    np.testing.assert_array_equal(apply_thresholds(["a"]*4, [-2.,-1.,1.,2.], learned), [0,0,1,1])


