"""Checks for statistical and counting mistakes that would invalidate the analysis."""
import numpy as np
import pytest

from Assignment_3.scripts.run_part2_analysis import classify_hybrid_errors, exact_sign_flip, holm_adjust


def test_five_consistent_wins_cannot_attain_five_percent():
    assert exact_sign_flip([1, 2, 3, 4, 5]) == .0625
    assert exact_sign_flip([-1, -2, -3, -4, -5]) == .0625
    assert exact_sign_flip([0, 0, 0, 0, 0]) == 1
    with pytest.raises(ValueError):
        exact_sign_flip([])


def test_holm_preserves_order_and_family_size():
    assert np.allclose(holm_adjust([.03, .001, .02]), [.04, .003, .04])
    assert np.allclose(holm_adjust([.8, .9, 1]), [1, 1, 1])


def test_categories_partition_all_binary_outcomes():
    gold = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    hybrid = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    dense = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    categories = classify_hybrid_errors(gold, hybrid, dense)
    assert np.array_equal(categories != "correct", gold != hybrid)
    assert categories.tolist() == ["correct", "correct", "FP fusion addition", "FP shared false match",
                                   "FN shared miss", "FN fusion loss", "correct", "correct"]
