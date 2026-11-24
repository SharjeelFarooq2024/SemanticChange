"""Official evaluation harness for SemEval-2027 Task 3, Subtask 2.

Provides:
- official_f1: Exactly matches vendored evaluation.py compute_f1_score.
- near_dup_groups: Connected-component near-duplicate grouping to eliminate text leakage.
- run_cv / repeated_cv: 5-fold x 5-repeat StratifiedGroupKFold on pooled OOF predictions.
- best_threshold: Inner out-of-fold calibration with shrinkage toward global cutoff.
- diagnostics: Per-word table of support, AUC, minority AP, oracle F1, actual F1, threshold loss, and minority recall.
- paired_bootstrap: Paired bootstrap significance testing on matched rows/repeats.
- leave_one_lemma_out: LOLO cross-validation across 10 lemmata.
- leave_one_period_out: LOPO cross-validation across 7 time periods.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================================ #
# 1. Authoritative Official Metric
# ============================================================================ #

def official_f1(
    words: Sequence[str],
    gold: Sequence[int],
    prediction: Sequence[int],
    per_word: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """Compute official Subtask 2 metric.
    
    Averages two-class macro F1 within each target word, then unweighted arithmetic
    mean across words. Exactly matches Assignment_3/vendor/official/evaluation.py.
    """
    words = np.asarray(words)
    gold = np.asarray(gold, dtype=int)
    prediction = np.asarray(prediction, dtype=int)

    if len(words) != len(gold) or len(gold) != len(prediction):
        raise ValueError("words, gold, and prediction must have identical length")

    by_word = defaultdict(list)
    for i, w in enumerate(words):
        by_word[w].append(i)

    word_scores: Dict[str, float] = {}
    for w in sorted(by_word.keys()):
        idx = by_word[w]
        g_w = gold[idx]
        p_w = prediction[idx]
        score = float(f1_score(g_w, p_w, average="macro", zero_division=0))
        word_scores[w] = score

    macro_f1 = float(np.mean(list(word_scores.values()))) if word_scores else 0.0

    if per_word:
        return macro_f1, word_scores
    return macro_f1


def fast_official_f1(words: np.ndarray, gold: np.ndarray, prediction: np.ndarray) -> float:
    """Vectorized official F1 equivalent using bincount for high-speed bootstrap/sweeps."""
    _, index = np.unique(words, return_inverse=True)
    size = index.max() + 1
    tp = np.bincount(index, weights=(gold == 1) & (prediction == 1), minlength=size)
    tn = np.bincount(index, weights=(gold == 0) & (prediction == 0), minlength=size)
    fp = np.bincount(index, weights=(gold == 0) & (prediction == 1), minlength=size)
    fn = np.bincount(index, weights=(gold == 1) & (prediction == 0), minlength=size)
    pden, nden = 2 * tp + fp + fn, 2 * tn + fp + fn
    pf = np.divide(2 * tp, pden, out=np.zeros(size), where=pden > 0)
    nf = np.divide(2 * tn, nden, out=np.zeros(size), where=nden > 0)
    n_classes = (pden > 0).astype(int) + (nden > 0).astype(int)
    return float(np.mean((pf + nf) / n_classes))


# ============================================================================ #
# 2. Near-Duplicate Grouping
# ============================================================================ #

def near_dup_groups(
    texts: Sequence[str],
    threshold: float = 0.80,
) -> np.ndarray:
    """Group near-duplicate texts into connected components.
    
    Identifies exact duplicates and near-duplicates via token-set Jaccard overlap
    (>= threshold) on normalized text. Returns integer group IDs for each text.
    """
    clean_texts = [
        re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(t).lower())).strip()
        for t in texts
    ]
    token_sets = [set(t.split()) for t in clean_texts]
    n = len(texts)

    # Inverted index for fast candidate filtering
    inverted: Dict[str, List[int]] = {}
    for i, s in enumerate(token_sets):
        for token in s:
            inverted.setdefault(token, []).append(i)

    # Candidate pairs sharing tokens
    pairs = set()
    for token, doc_ids in inverted.items():
        if 1 < len(doc_ids) < 300:
            for idx_a in range(len(doc_ids)):
                for idx_b in range(idx_a + 1, len(doc_ids)):
                    i, j = doc_ids[idx_a], doc_ids[idx_b]
                    if i > j:
                        i, j = j, i
                    pairs.add((i, j))

    # Also check exact normalized text match
    text_to_indices: Dict[str, List[int]] = {}
    for i, ct in enumerate(clean_texts):
        text_to_indices.setdefault(ct, []).append(i)
    for idx_list in text_to_indices.values():
        if len(idx_list) > 1:
            for idx_a in range(len(idx_list)):
                for idx_b in range(idx_a + 1, len(idx_list)):
                    i, j = sorted((idx_list[idx_a], idx_list[idx_b]))
                    pairs.add((i, j))

    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i, j in pairs:
        s1, s2 = token_sets[i], token_sets[j]
        if not s1 or not s2:
            if clean_texts[i] == clean_texts[j]:
                adj[i].append(j)
                adj[j].append(i)
            continue
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        if union > 0 and (intersection / union >= threshold or clean_texts[i] == clean_texts[j]):
            adj[i].append(j)
            adj[j].append(i)

    # Connected components
    visited = set()
    groups = np.zeros(n, dtype=int)
    group_id = 0
    for i in range(n):
        if i not in visited:
            queue = [i]
            visited.add(i)
            comp = [i]
            while queue:
                curr = queue.pop(0)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        comp.append(neighbor)
                        queue.append(neighbor)
            for member in comp:
                groups[member] = group_id
            group_id += 1

    return groups


# ============================================================================ #
# 3. Repeated Stratified Group Cross-Validation
# ============================================================================ #

def run_cv(
    data: pd.DataFrame,
    model_factory: Callable[[int], Any],
    n_splits: int = 5,
    n_repeats: int = 5,
    base_seed: int = 42,
    group_col: str = "group",
    strat_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute 5-fold x 5-repeat StratifiedGroupKFold on pooled out-of-fold predictions.
    
    Stratification is on (word x label), grouped by near_dup_groups.
    Metrics are computed on pooled out-of-fold predictions across all rows for each repeat,
    then averaged over repeats.
    """
    df = data.reset_index(drop=True).copy()
    n_samples = len(df)
    labels = df["label"].to_numpy().astype(int)
    words = df["word"].to_numpy().astype(str)

    if strat_col is None:
        strat_keys = df["word"].astype(str) + "_" + df["label"].astype(str)
    else:
        strat_keys = df[strat_col].astype(str)

    groups = df[group_col].to_numpy()

    repeat_official_f1 = []
    repeat_pooled_macro_f1 = []
    all_oof_preds = np.zeros((n_repeats, n_samples), dtype=int)
    all_oof_probs = np.zeros((n_repeats, n_samples), dtype=float)

    for r in range(n_repeats):
        seed = base_seed + r * 17
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        oof_pred = np.full(n_samples, -1, dtype=int)
        oof_prob = np.full(n_samples, np.nan, dtype=float)

        for fold, (train_idx, test_idx) in enumerate(splitter.split(df, strat_keys, groups)):
            # Verify no group overlap
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            overlap = train_groups & test_groups
            if overlap:
                raise AssertionError(f"Group leakage detected in repeat {r} fold {fold}: {len(overlap)} groups")

            train_fold = df.iloc[train_idx].copy()
            test_fold = df.iloc[test_idx].copy()

            fold_seed = seed + fold
            model = model_factory(fold_seed)
            model.fit(train_fold)

            preds = model.predict(test_fold)
            oof_pred[test_idx] = preds

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(test_fold)
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    oof_prob[test_idx] = probs[:, 1]
                else:
                    oof_prob[test_idx] = probs.ravel()
            elif hasattr(model, "decision_function"):
                margins = model.decision_function(test_fold)
                # sigmoid
                oof_prob[test_idx] = 1.0 / (1.0 + np.exp(-margins))
            else:
                oof_prob[test_idx] = preds.astype(float)

        if (oof_pred == -1).any():
            raise AssertionError(f"Incomplete OOF predictions in repeat {r}")

        # Compute official metric on pooled OOF predictions for this repeat
        score = official_f1(words, labels, oof_pred)
        pooled_macro = float(f1_score(labels, oof_pred, average="macro", zero_division=0))

        repeat_official_f1.append(score)
        repeat_pooled_macro_f1.append(pooled_macro)
        all_oof_preds[r] = oof_pred
        all_oof_probs[r] = oof_prob

    mean_f1 = float(np.mean(repeat_official_f1))
    std_f1 = float(np.std(repeat_official_f1, ddof=1)) if len(repeat_official_f1) > 1 else 0.0

    return {
        "mean_official_f1": mean_f1,
        "std_official_f1": std_f1,
        "repeat_official_f1": repeat_official_f1,
        "mean_pooled_macro_f1": float(np.mean(repeat_pooled_macro_f1)),
        "oof_predictions": all_oof_preds,
        "oof_probabilities": all_oof_probs,
    }


repeated_cv = run_cv


# ============================================================================ #
# 4. Calibration & Best Threshold
# ============================================================================ #

def _word_macro_f1_for_thresh(y: np.ndarray, prob: np.ndarray, t: float) -> float:
    pred = (prob >= t).astype(int)
    return float(f1_score(y, pred, average="macro", zero_division=0))


def best_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    words: Sequence[str],
    shrinkage: float = 0.5,
    n_quantiles: int = 39,
) -> Dict[str, Any]:
    """Fit per-word decision thresholds shrunk toward a global cutoff.
    
    threshold(w) = shrinkage * global_threshold + (1 - shrinkage) * local_threshold(w)
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    w = np.asarray(words)

    candidate_thresholds = np.linspace(0.05, 0.95, n_quantiles)

    # 1. Global cutoff maximizing overall macro F1
    best_global_t = 0.5
    best_global_score = -1.0
    for t in candidate_thresholds:
        score = official_f1(w, y, (p >= t).astype(int))
        if score > best_global_score:
            best_global_score = score
            best_global_t = float(t)

    # 2. Local cutoffs per word
    calibrated_thresholds: Dict[str, float] = {}
    for word_val in np.unique(w):
        mask = w == word_val
        y_w, p_w = y[mask], p[mask]

        best_w_t = best_global_t
        best_w_score = -1.0

        # Only sweep if both classes exist
        if len(np.unique(y_w)) >= 2:
            for t in candidate_thresholds:
                # Safeguard: do not allow trivial 0 or all 1 prediction if both classes exist
                pred_w = (p_w >= t).astype(int)
                if len(np.unique(pred_w)) < 2:
                    continue
                score = _word_macro_f1_for_thresh(y_w, p_w, t)
                if score > best_w_score:
                    best_w_score = score
                    best_w_t = float(t)

        final_t = shrinkage * best_global_t + (1.0 - shrinkage) * best_w_t
        calibrated_thresholds[str(word_val)] = float(final_t)

    return {
        "global_threshold": best_global_t,
        "per_word": calibrated_thresholds,
        "shrinkage": shrinkage,
    }


def apply_calibrated_thresholds(
    words: Sequence[str],
    probs: Sequence[float],
    calibration: Dict[str, Any],
) -> np.ndarray:
    w = np.asarray(words)
    p = np.asarray(probs, dtype=float)
    thresholds = calibration["per_word"]
    global_t = calibration.get("global_threshold", 0.5)

    cutoffs = np.array([thresholds.get(str(word_val), global_t) for word_val in w])
    return (p >= cutoffs).astype(int)


# ============================================================================ #
# 5. Diagnostics Table
# ============================================================================ #

def diagnostics(
    data: pd.DataFrame,
    y_pred: Sequence[int],
    y_prob: Sequence[float],
) -> pd.DataFrame:
    """Generate per-word diagnostics table.
    
    Columns: word, total_support, minority_class, minority_support,
             auc, minority_ap, oracle_f1, actual_f1, threshold_loss, minority_recall.
    """
    df = data.copy()
    y_true = df["label"].to_numpy().astype(int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    words = df["word"].to_numpy().astype(str)

    rows = []
    candidate_thresholds = np.linspace(0.01, 0.99, 99)

    for w in sorted(np.unique(words)):
        mask = words == w
        y_w = y_true[mask]
        pred_w = y_pred[mask]
        prob_w = y_prob[mask]

        total_n = len(y_w)
        counts = Counter(y_w)
        c0, c1 = counts.get(0, 0), counts.get(1, 0)

        min_class = 1 if c1 <= c0 else 0
        min_support = min(c0, c1)

        # AUC
        if len(np.unique(y_w)) >= 2:
            auc = float(roc_auc_score(y_w, prob_w))
            # Minority AP
            y_min = (y_w == min_class).astype(int)
            p_min = prob_w if min_class == 1 else (1.0 - prob_w)
            min_ap = float(average_precision_score(y_min, p_min))
        else:
            auc = 1.0
            min_ap = 1.0

        # Actual F1
        actual_f1 = float(f1_score(y_w, pred_w, average="macro", zero_division=0))

        # Oracle F1 (best threshold for this word)
        oracle_f1 = actual_f1
        if len(np.unique(y_w)) >= 2:
            for t in candidate_thresholds:
                t_pred = (prob_w >= t).astype(int)
                score = float(f1_score(y_w, t_pred, average="macro", zero_division=0))
                if score > oracle_f1:
                    oracle_f1 = score

        threshold_loss = float(oracle_f1 - actual_f1)

        # Minority Recall
        min_mask = y_w == min_class
        if min_support > 0:
            min_rec = float(np.mean(pred_w[min_mask] == min_class))
        else:
            min_rec = 1.0

        rows.append({
            "word": w,
            "total_support": int(total_n),
            "minority_class": int(min_class),
            "minority_support": int(min_support),
            "unstable_flag": bool(min_support < 10),
            "auc": float(auc),
            "minority_ap": float(min_ap),
            "oracle_f1": float(oracle_f1),
            "actual_f1": float(actual_f1),
            "threshold_loss": float(threshold_loss),
            "minority_recall": float(min_rec),
        })

    return pd.DataFrame(rows)


# ============================================================================ #
# 6. Paired Bootstrap Hypothesis Testing
# ============================================================================ #

def paired_bootstrap(
    data: pd.DataFrame,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap comparison between system A and system B.
    
    preds_a, preds_b: shape (n_repeats, n_samples)
    Computes distribution of delta = Score_A - Score_B.
    Adoption rule: lower bound of 95% CI > 0.
    """
    words = data["word"].to_numpy().astype(str)
    labels = data["label"].to_numpy().astype(int)
    n_repeats, n_samples = preds_a.shape

    # Compute base mean scores across repeats
    scores_a = [official_f1(words, labels, preds_a[r]) for r in range(n_repeats)]
    scores_b = [official_f1(words, labels, preds_b[r]) for r in range(n_repeats)]

    mean_a = float(np.mean(scores_a))
    mean_b = float(np.mean(scores_b))
    point_diff = float(mean_a - mean_b)

    rng = np.random.default_rng(seed)

    boot_diffs = []
    # Resample instances with replacement per repeat
    for _ in range(n_boot):
        sample_indices = rng.integers(0, n_samples, size=n_samples)
        b_words = words[sample_indices]
        b_labels = labels[sample_indices]

        # Check if all 10 words exist in bootstrap sample
        if len(np.unique(b_words)) < len(np.unique(words)):
            continue

        b_diffs_repeat = []
        for r in range(n_repeats):
            pa = preds_a[r, sample_indices]
            pb = preds_b[r, sample_indices]
            sa = fast_official_f1(b_words, b_labels, pa)
            sb = fast_official_f1(b_words, b_labels, pb)
            b_diffs_repeat.append(sa - sb)

        boot_diffs.append(np.mean(b_diffs_repeat))

    boot_diffs = np.asarray(boot_diffs)
    ci_lo = float(np.quantile(boot_diffs, 0.025))
    ci_hi = float(np.quantile(boot_diffs, 0.975))
    p_not_better = float(np.mean(boot_diffs <= 0.0))

    adopted = bool(ci_lo > 0.0)

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "point_diff": point_diff,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "p_a_not_better": p_not_better,
        "adopted": adopted,
        "n_bootstrap": len(boot_diffs),
    }


# ============================================================================ #
# 7. Generalization Checks: LOLO & LOPO
# ============================================================================ #

def leave_one_lemma_out(
    data: pd.DataFrame,
    model_factory: Callable[[int], Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """Leave-One-Lemma-Out (LOLO) evaluation across the 10 target words."""
    df = data.reset_index(drop=True).copy()
    words = sorted(df["word"].unique())
    n_samples = len(df)

    oof_preds = np.full(n_samples, -1, dtype=int)
    lemma_scores = {}

    for i, held_out in enumerate(words):
        train_idx = np.where(df["word"] != held_out)[0]
        test_idx = np.where(df["word"] == held_out)[0]

        train_data = df.iloc[train_idx].copy()
        test_data = df.iloc[test_idx].copy()

        model = model_factory(seed + i)
        model.fit(train_data)

        preds = model.predict(test_data)
        oof_preds[test_idx] = preds

        w_f1 = float(f1_score(test_data["label"], preds, average="macro", zero_division=0))
        lemma_scores[held_out] = w_f1

    mean_score = float(np.mean(list(lemma_scores.values())))
    return {
        "mean_lolo_f1": mean_score,
        "per_lemma_f1": lemma_scores,
        "predictions": oof_preds,
    }


def leave_one_period_out(
    data: pd.DataFrame,
    model_factory: Callable[[int], Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """Leave-One-Period-Out (LOPO) evaluation across historical time periods."""
    df = data.reset_index(drop=True).copy()
    periods = sorted(df["period_label"].unique())
    n_samples = len(df)

    oof_preds = np.full(n_samples, -1, dtype=int)
    period_scores = {}

    for i, held_out in enumerate(periods):
        train_idx = np.where(df["period_label"] != held_out)[0]
        test_idx = np.where(df["period_label"] == held_out)[0]

        train_data = df.iloc[train_idx].copy()
        test_data = df.iloc[test_idx].copy()

        model = model_factory(seed + i)
        model.fit(train_data)

        preds = model.predict(test_data)
        oof_preds[test_idx] = preds

        score = official_f1(test_data["word"], test_data["label"], preds)
        period_scores[held_out] = score

    mean_score = float(np.mean(list(period_scores.values())))
    return {
        "mean_lopo_f1": mean_score,
        "per_period_f1": period_scores,
        "predictions": oof_preds,
    }
