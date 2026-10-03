# Baseline Phase Implementation

This document describes the implemented baseline phase for the Swedish
development data. It is a code-oriented guide: the descriptions below follow
the current modules under `src/baselines`, the configuration in
`configs/baselines.yaml`, and the artifacts written under
`outputs/baselines`.

## 1. Purpose and Scope

The baseline phase establishes reproducible reference systems for both task
subtasks:

- **Subtask 1: Diachronic Word Sense Induction (WSI).** Cluster usages of
  each target word without using labels during clustering, then compare the
  induced clusters with gold singleton sense labels.
- **Subtask 2: Binary usage classification.** Predict the binary target-sense
  label for each usage.

Two representation families are implemented:

1. **Baseline A: Classical models**, using word and character TF-IDF,
   hand-built EDA text features, categorical features, and KMeans.
2. **Baseline B: Frozen XLM-R**, using mean-pooled
   `FacebookAI/xlm-roberta-base` embeddings as fixed features for linear
   classifiers and KMeans.

The raw files in `data/raw/dev/SV` are read but never modified. All paths,
model settings, feature lists, and output locations are controlled by
`configs/baselines.yaml`.

## 2. Execution Entry Points

Run commands from the repository root:

```powershell
python -m src.baselines.baseline_a_subtask2
python -m src.baselines.baseline_a_subtask1
python -m src.baselines.pretrained_subtask2
python -m src.baselines.pretrained_subtask1
python -m src.baselines.evaluate
python -m pytest tests/ -v
```

The classical commands can run with the standard scientific Python
dependencies. The pretrained commands additionally require `torch`,
`transformers`, access to the XLM-R weights, and enough disk space for the
model and embedding cache.

The commands are independent. A typical complete run is:

1. Run classical Subtask 2 and Subtask 1.
2. Run pretrained Subtask 2 and Subtask 1 after model weights are available.
3. Run `evaluate` to combine available outputs and generate diagnostics.

The evaluator handles missing classical or pretrained output suites without
failing; it records missing suites in `evaluation_diagnostics.json`.

## 3. Configuration

`configs/baselines.yaml` contains the following important settings:

| Area | Implemented setting |
|---|---|
| Reproducibility | Seed `42` |
| Classification split | 80/20, random state `42`, stratified by label |
| Word TF-IDF | Unigrams and bigrams, sublinear TF, `min_df=1`, max 50,000 features |
| Character TF-IDF | Character 3--5 grams, sublinear TF, max 50,000 features |
| Logistic regression | `max_iter=2000`, balanced class weights |
| Random Forest | 100 trees, maximum depth 15, balanced class weights |
| Subtask 1 clustering | KMeans, requested `k=3`, `n_init=10` |
| Frozen encoder | `FacebookAI/xlm-roberta-base`, mean pooling, max length 512, batch size 32 |
| Output roots | `outputs/baselines/{classical,pretrained,figures,tables}` |

`src.baselines.common.load_config` resolves the project root from the caller
or from the source tree and adds it to the loaded configuration.

## 4. Shared Data Preparation

### 4.1 Loading and normalization

`load_baseline_bundle` verifies that the four configured input files exist:

- `usages.jsonl`
- `subtask1.jsonl`
- `subtask2.jsonl`
- `definitions.jsonl`

It delegates parsing to `src.eda.load_data.build_dataset_bundle`. Usage rows
are normalized to include `word`, `sentence_id`, `text`, `pos`,
`period_label`, character offsets, `year`, `year_int`, and a derived
`usage_id`. Missing optional columns are created with empty or missing values.

### 4.2 Subtask 2 label join

`join_subtask2_labels` joins usage records to labels on `sentence_id`.
Labels are accepted only when they normalize to integer `0` or `1`; booleans
are converted to integers and malformed labels are discarded. Duplicate label
IDs are counted and the first valid row is retained.

The function returns the matched data and diagnostics containing:

- usage and label record counts;
- matched and unmatched usage records;
- malformed labels;
- duplicate label records.

The downstream classifiers use the resulting `label_value` column. A run is
stopped if the joined data contains fewer than two classes.

### 4.3 Subtask 1 label join

`baseline_a_subtask1._join_subtask1` joins on the pair
`(sentence_id, word)`. `parse_sense_labels` preserves list-valued annotations
instead of forcing them into one arbitrary class. A singleton annotation is
stored as `singleton_sense`; multi-valued annotations receive a missing
singleton value and remain available in `sense_labels`.

The join records malformed or missing labels, multi-label records, duplicate
keys, unmatched usage records, and matched records. Duplicate keys retain the
first row.

## 5. Classical Feature Construction

### 5.1 EDA text features

`classical_features.build_feature_frame` calls
`src.eda.text_features.add_text_features` while preserving the original usage
rows. The configured numeric features are:

1. `text_length_chars`
2. `token_count`
3. `unique_token_count`
4. `type_token_ratio`
5. `avg_token_length`
6. `punctuation_ratio`
7. `digit_ratio`
8. `stopword_ratio`
9. `target_position`

The feature implementation uses a lightweight Swedish-aware tokenizer and a
small Swedish stopword list. It also derives `lexical_diversity` and
`target_in_text`, although those two columns are not part of the configured
classical numeric feature list.

### 5.2 Categorical features

The full interpretable representation one-hot encodes:

- `word`;
- `pos`;
- `period_label`.

Unknown categories at evaluation time are ignored by
`OneHotEncoder(handle_unknown="ignore")`.

### 5.3 Preprocessors and leakage control

`build_interpretable_preprocessor` returns a `ColumnTransformer` with three
branches:

- word-level TF-IDF over `text`;
- standardized numeric EDA features;
- one-hot categorical features.

`build_ablated_preprocessor` has the same structure but excludes categorical
`word`. These preprocessors are placed inside sklearn `Pipeline` objects in
the classification runner. Consequently, TF-IDF vocabularies, means and
scales, and category vocabularies are fitted only on the training split.

## 6. Classical Baseline A: Subtask 2

`baseline_a_subtask2.run` performs the following steps:

1. Load configuration and seed Python and NumPy.
2. Create output directories.
3. Load and validate the data bundle.
4. Join Subtask 2 labels.
5. Create one stratified 80/20 split using `random_state=42`.
6. Fit each model on the training portion and predict the held-out portion.
7. Save metrics, predictions, metadata, and interpretability artifacts.

The implemented models are:

| Model ID | Representation and estimator |
|---|---|
| `majority` | `DummyClassifier(strategy="most_frequent")` |
| `tfidf_logistic_regression` | Word 1--2 gram TF-IDF + balanced Logistic Regression |
| `word_char_tfidf_linear_svm` | Word 1--2 gram and character 3--5 gram TF-IDF via `FeatureUnion` + balanced `LinearSVC` |
| `tfidf_complement_nb` | Word 1--2 gram TF-IDF + `ComplementNB(norm=True)` |
| `eda_features_logistic_regression` | TF-IDF + numeric EDA + categorical features + balanced Logistic Regression |
| `eda_features_random_forest` | Same combined features + balanced Random Forest |
| `ablated_no_word_id_logistic_regression` | Masked target token, no categorical `word`, combined features + balanced Logistic Regression |

### 6.1 Word-identity ablation

For the ablated model, every case-insensitive, word-boundary match of the
target word in the text is replaced with `[TARGET]`. The categorical `word`
column is also removed. This provides a comparison between a model that can
memorize target-word base rates and a model restricted to context, structure,
part of speech, and period.

### 6.2 Classification metrics

For every model, `_metrics` writes:

- accuracy;
- macro-F1;
- weighted-F1;
- macro precision;
- macro recall;
- balanced accuracy.

Macro-F1 is the primary comparison metric because the development labels are
imbalanced. Balanced accuracy is the secondary imbalance-aware metric.

### 6.3 Interpretability outputs

For the two Logistic Regression models with accessible coefficients, the
runner saves the 25 largest positive and 25 largest negative coefficients.
For the Random Forest, it saves the top 100 Gini importances. PNG plots of
the leading coefficients and importances are also written to the classical
output directory.

## 7. Classical Baseline A: Subtask 1

`baseline_a_subtask1.run` performs unsupervised clustering independently for
each target word:

1. Load and join usage and Subtask 1 labels.
2. Group all matched usages by `word`.
3. Fit a word-level TF-IDF vectorizer separately for each word.
4. Add standardized numeric EDA features when
   `use_lexical_features=true`.
5. Horizontally concatenate TF-IDF and numeric features.
6. Fit KMeans with three requested clusters, `n_init=10`, and seed `42`.
7. Store the cluster assignment for every usage.

The actual cluster count is `min(requested_clusters, number_of_usages)`. A
one-row group receives cluster `0` without fitting KMeans.

Clustering uses the complete usage set for each word; there is no
classification train/test split in this subtask. Gold labels are used only
after clustering for evaluation.

### 7.1 Subtask 1 metrics

`_valid_clustering_metrics` evaluates only rows with a singleton gold sense:

- **ARI:** chance-corrected agreement between clusters and gold senses;
- **NMI:** normalized information overlap;
- **Purity:** fraction of records assigned to the dominant gold sense within
  each cluster.

Multi-label rows are preserved and counted in
`multi_label_records_excluded`, but they do not enter ARI, NMI, or purity.
Metrics are returned as missing when there are not at least two gold senses
and two predicted clusters.

The runner also writes cluster proportions by period for both predicted
clusters and singleton gold senses. A change in the dominant cluster across
periods is exploratory only: it can result from genre, corpus source,
topic, sampling, or annotation effects rather than semantic change.

## 8. Frozen XLM-R Baseline B

### 8.1 Context and embedding cache

`load_frozen_usage_embeddings` builds one context per usage with
`embedding_analysis.build_usage_context`. When the target word occurs in the
text, the context is a window extending up to 80 characters on either side;
otherwise the full text is used.

`encode_texts` tokenizes batches, truncates to 512 tokens, runs the model in
evaluation mode under `torch.no_grad()`, and mean-pools final hidden states
using the attention mask. The resulting vectors are 768-dimensional for the
configured base model.

The cache consists of:

- `data/processed/eda/embeddings/usage_embeddings.pkl`;
- `data/processed/eda/embeddings/usage_embeddings_metadata.json`.

The cache is reused only when its record fingerprint, model name, maximum
length, pooling mode, schema, row count, and saved contexts all match the
current request. The fingerprint is SHA-256 over usage identity fields and
the generated context strings. Invalid or incomplete caches are regenerated.

### 8.2 Frozen Subtask 2

`pretrained_subtask2.run` joins labels to the cached embeddings, applies the
same stratified split as the classical classifier, and fits two downstream
models:

- `frozen_xlmr_logistic_regression`;
- `frozen_xlmr_linear_svm`.

Each model standardizes the embedding vectors inside a pipeline and uses
balanced class weights. The XLM-R encoder itself is never updated and never
receives labels.

### 8.3 Frozen Subtask 1

`pretrained_subtask1.run` joins labels to cached embeddings and applies KMeans
per target word with the configured cluster count, `n_init=10`, and seed `42`.
Unlike the classical Subtask 1 implementation, it clusters the embedding
matrix directly and does not append the nine numeric EDA features. It uses
the same singleton-only metric policy and period-distribution output.

## 9. Unified Evaluation

`evaluate.run` consolidates whatever baseline outputs are present.

### 9.1 Subtask 2 consolidation

It reads classical and pretrained `subtask2_metrics.csv` files, normalizes
minor schema differences, concatenates them, and ranks models by macro-F1.
It also reads prediction files to produce:

- per-word metrics;
- per-period metrics;
- confusion matrices for each model.

The per-group table is diagnostic, not a replacement for the fixed held-out
test metric. Small groups can have unstable scores, and word-level scores can
reflect very different label base rates.

### 9.2 Subtask 1 consolidation

It combines classical and pretrained metrics, assignments, and period
distributions. The generated temporal interpretation reports changes in the
dominant predicted cluster by word and explicitly warns that these changes do
not prove semantic drift.

### 9.3 Data-quality diagnostics and findings

The evaluator checks input availability, label joins, period-label format,
duplicate usage text, prediction coverage, and the configured leakage design.
It writes `evaluation_diagnostics.json` and a dynamic
`baseline_findings.md` using a Read--Reason--Interpret--Verify structure.

## 10. Output Artifacts

### Classical and pretrained model directories

Each model suite can write:

- `subtask2_metrics.csv`;
- `subtask2_predictions.csv`;
- `subtask2_metadata.json`;
- `subtask1_metrics.csv`;
- `subtask1_cluster_assignments.csv`;
- `subtask1_period_distributions.csv`;
- Subtask 1 metadata and plots where implemented.

The classical Subtask 2 runner additionally writes coefficient and Random
Forest importance CSV files and PNG plots.

### Unified tables and figures

The evaluator writes:

- `subtask2_comparison.csv`;
- `subtask2_model_ranking.csv`;
- `subtask2_per_group.csv`;
- `subtask1_comparison.csv`;
- `subtask1_cluster_period_distributions.csv`;
- `subtask1_cluster_assignments.csv`;
- `subtask1_temporal_interpretation.md`;
- `evaluation_diagnostics.json`;
- `baseline_findings.md`;
- `confusion_matrix_*.png`.

## 11. Reproducibility and Validation

The test suite validates the baseline foundations rather than retraining every
model on every test run. It covers:

- expected input bundle availability;
- malformed and unmatched Subtask 2 joins;
- preservation of multi-valued Subtask 1 labels;
- deterministic per-word clustering;
- unified metric schema and rankability;
- reuse of a valid embedding cache;
- a clear error when the pretrained model cannot be loaded;
- presence of EDA features;
- exclusion of `word` from the ablated preprocessor;
- deterministic TF-IDF behavior.

The main reproducibility controls are the fixed seed, explicit split random
state, deterministic KMeans settings, train-fitted sklearn pipelines, frozen
encoder parameters, and validated embedding fingerprints.

## 12. Interpretation Limits and Implementation Notes

The strongest classical Subtask 2 score is not automatically evidence of
semantic-change understanding. The full model can exploit target-word label
prevalence through the categorical `word` feature and lexical context. The
word-identity ablation and per-word/per-period tables are therefore essential
for interpreting results.

Similarly, high clustering purity can be caused by imbalanced gold senses,
and temporal cluster shifts can be caused by corpus composition. ARI, NMI,
purity, temporal distributions, and the ablation should be read together.

The generated evaluator tables may contain both classical and pretrained rows
for the same target word. Counts in generated findings therefore depend on
which suites have been run. The source report and generated artifacts should
be checked against the current output directory rather than treated as a
permanent snapshot of one execution.
