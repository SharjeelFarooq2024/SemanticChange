# Evaluation and modeling notes

## Task formulation
This is a diachronic semantic-change task. The input is a contextual usage of a target word in a historical or contemporary corpus segment, and the output is a label related to meaning or usage change across temporal slices. The project uses both word-level and usage-level signals, which makes the task naturally compatible with supervised classification.

## Likely label structure
- Subtask 1: sense labels or sense-group assignments across time periods
- Subtask 2: binary usage labels for whether a usage fits a target sense or usage class

## Why the metric matters
Because the dataset is imbalanced across target words and time periods, a naive accuracy score can look high while hiding weak performance on minority classes. For this reason, the downstream evaluation should prioritize macro-F1 or another class-balanced metric rather than raw accuracy alone.

## Interpretation of the metric choice
- Macro-F1 rewards balanced performance across classes
- It is safer than micro-F1 when minority senses matter
- It helps highlight whether the model is truly detecting semantic change rather than simply exploiting frequent periods or common target words

## Data caveats that affect evaluation
- Some target words are much more frequent than others
- Temporal coverage is uneven across periods
- Target offsets are not always exact, so noisy context boundaries may affect feature extraction and final predictions
- The context itself can contain source effects unrelated to semantic shift

## Practical implication
The best immediate baseline should be evaluated with class-balanced metrics and then compared against a more contextual model. This keeps the comparison fair before moving to advanced modeling.
