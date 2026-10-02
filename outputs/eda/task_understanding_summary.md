# Task understanding summary

## Black-box task view
This project targets semantic change across multiple time periods in Swedish texts. The raw data contains usage examples for target words, their period labels, and context sentences. The task is to understand whether a target word changes meaning or usage patterns across time periods, and to support classification-style downstream modeling such as sense labeling or binary usage detection.

## White-box signals
The likely cues are lexical context, target-word frequency, POS patterns, temporal distribution, and contextual embedding structure. Reliable EDA therefore focuses on what is observable in the text rather than assuming a final model architecture.

## Output and label space
- Input: contextual text snippets with a target word, period label, and sentence metadata
- Output: a semantic change signal or sense-related classification label, depending on subtask
- Subtask 1 is sense-related and label-bearing; Subtask 2 is binary usage labeling at the sentence/usage level
- The dataset already contains both word-level and usage-level labels, which suggests a supervised or weakly supervised modeling pipeline downstream

## What makes the task difficult
The main challenges are: period imbalance, word imbalance, annotation noise in target offsets, and the fact that apparent semantic drift can be confused with corpus composition or source effects. This is why selective EDA is important before moving to any baseline.

## Evaluation intuition
The task is classification-like rather than pure unsupervised clustering. In a downstream pipeline, macro-F1 is a reasonable default target because the word and period distributions are imbalanced. The exact official metric should still be confirmed from the task description when the final reporting step is performed.
