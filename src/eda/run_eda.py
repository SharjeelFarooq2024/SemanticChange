from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from .dataset_stats import compute_dataset_statistics, write_dataset_tables
from .embedding_analysis import analyze_embeddings
from .load_data import build_dataset_bundle
from .plots import (
    build_eda_pdf,
    plot_context_length,
    plot_dataset_composition,
    plot_definition_usage_similarity,
    plot_embedding_structure,
    plot_lexical_characteristics,
    plot_temporal_distribution,
    plot_temporal_embedding_similarity,
    plot_word_usage_distribution,
)
from .text_features import add_text_features


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_findings(stats: Dict[str, Any], sim_df: pd.DataFrame, def_sim_df: pd.DataFrame) -> str:
    summary = stats["summary"]
    word_stats = stats["word_statistics"]
    offset_quality = stats["offset_quality"].set_index("quality_label")["count"].to_dict()
    period_stats = stats["period_statistics"]
    max_word = word_stats.iloc[0]["word"] if not word_stats.empty else "N/A"
    max_count = int(word_stats.iloc[0]["usage_count"]) if not word_stats.empty else 0
    min_count = int(word_stats.iloc[-1]["usage_count"]) if not word_stats.empty else 0
    period_max = period_stats["usage_count"].max() if not period_stats.empty else 0
    period_min = period_stats["usage_count"].min() if not period_stats.empty else 0
    avg_sim = float(sim_df["similarity"].mean()) if not sim_df.empty else 0.0
    avg_def_sim = float(def_sim_df["cosine_similarity"].mean()) if not def_sim_df.empty else 0.0

    text = [
        "# EDA Findings",
        "",
        "## A. Data Quality",
        "",
        "### Observation",
        f"The official Swedish development set contains {summary['usage_records']} usage records across {summary['unique_target_words']} target words and {summary['unique_periods']} periods. The offset-quality check reports exact matches in {offset_quality.get('exact target match', 0)} cases, plausible inflections/derivations in {offset_quality.get('plausible inflection/derivation', 0)} cases, shared-stem candidates in {offset_quality.get('shared-stem candidate', 0)} cases, and suspicious mismatches in {offset_quality.get('suspicious/mismatch', 0)} cases.",
        "",
        "### Interpretation",
        "The data is usable for EDA, but the target-span metadata is not uniformly reliable. This means target-aware representation extraction must be conservative and should not silently replace the original offsets.",
        "",
        "### Modeling Implication",
        "The eventual system should prefer context-level or target-window representations that are robust to noisy offsets, and should explicitly track alignment uncertainty rather than assuming the original span annotations are perfect.",
        "",
        "## B. Temporal Distribution",
        "",
        "### Observation",
        f"The most active period contains {period_max} usages and the least active period contains {period_min}. The dataset is not perfectly balanced across time slices, which can create spurious apparent drift when periods differ in corpus composition.",
        "",
        "### Interpretation",
        "Temporal imbalance is an important confound because apparent word change may reflect corpus breadth or editorial source shifts rather than actual semantic change.",
        "",
        "### Modeling Implication",
        "Any eventual temporal analysis should consider minimum-support filtering, period normalization, and explicit checks for corpus-composition confounds before interpreting changes as semantic drift.",
        "",
        "## C. Target-Word Imbalance",
        "",
        "### Observation",
        f"Target words are unevenly represented: the most frequent word is {max_word} with {max_count} usages, while the least frequent target has {min_count} usages.",
        "",
        "### Interpretation",
        "This imbalance can distort clustering, temporal comparisons, and training dynamics if some words dominate the evaluation assumptions.",
        "",
        "### Modeling Implication",
        "Per-word normalization, weighting, or minimum-support handling should be considered when building downstream baselines and temporal change estimates.",
        "",
        "## D. Embedding Structure",
        "",
        "### Observation",
        f"The average adjacent-period embedding similarity is {avg_sim:.3f}, and the average definition-to-usage cosine similarity is {avg_def_sim:.3f}. This suggests there is a stable but not necessarily purely semantic organizing structure in contextual space.",
        "",
        "### Interpretation",
        "Embedding-space separation may reflect target word identity, period-driven corpus shift, or lexical context differences rather than genuine sense boundaries.",
        "",
        "### Modeling Implication",
        "Clusters should be treated as exploratory evidence, not as direct sense labels. Downstream modeling should evaluate whether separation is better explained by target word, period, POS, or gold labels before making stronger claims.",
        "",
    ]
    return "\n".join(text) + "\n"


def run_eda() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs" / "eda.yaml"
    config = load_config(config_path)
    output_dir = root / config["paths"]["outputs_dir"]
    figures_dir = root / config["paths"]["figures_dir"]
    tables_dir = root / config["paths"]["tables_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_dataset_bundle(root)
    usage_df = bundle["usage"]
    subtask1_df = bundle["subtask1"]
    subtask2_df = bundle["subtask2"]
    definitions_df = bundle["definitions"]

    stats = compute_dataset_statistics(usage_df, subtask1_df, subtask2_df, definitions_df)
    usage_df = stats["usage_df"]
    tables = write_dataset_tables(tables_dir, stats)
    feature_df = add_text_features(usage_df)

    fig_paths = {
        "01_dataset_composition": plot_dataset_composition(feature_df, figures_dir / "01_dataset_composition.png"),
        "02_temporal_distribution": plot_temporal_distribution(feature_df, figures_dir / "02_temporal_distribution.png"),
        "03_usages_per_target_word": plot_word_usage_distribution(feature_df, figures_dir / "03_usages_per_target_word.png"),
        "04_context_length": plot_context_length(feature_df, figures_dir / "04_context_length.png"),
        "05_lexical_characteristics": plot_lexical_characteristics(feature_df, figures_dir / "05_lexical_characteristics.png"),
    }

    embedding_analysis = analyze_embeddings(feature_df, config, definitions_df, subtask2_df)
    pca_df = embedding_analysis["pca"]
    temporal_similarity = embedding_analysis["temporal_similarity"]
    definition_similarity = embedding_analysis.get("definition_usage_similarity", pd.DataFrame())

    if not temporal_similarity.empty:
        temporal_similarity.to_csv(tables_dir / "temporal_similarity.csv", index=False)
    if not definition_similarity.empty:
        definition_similarity.to_csv(tables_dir / "definition_usage_similarity.csv", index=False)

    fig_paths["06_embedding_structure"] = plot_embedding_structure(pca_df, figures_dir / "06_embedding_structure.png")
    fig_paths["07_temporal_embedding_similarity"] = plot_temporal_embedding_similarity(temporal_similarity, figures_dir / "07_temporal_embedding_similarity.png")
    fig_paths["08_definition_usage_similarity"] = plot_definition_usage_similarity(definition_similarity, figures_dir / "08_definition_usage_similarity.png")

    findings_md = build_findings(stats, temporal_similarity, definition_similarity)
    findings_path = root / config["paths"]["findings_path"]
    findings_path.parent.mkdir(parents=True, exist_ok=True)
    findings_path.write_text(findings_md, encoding="utf-8")

    pdf_path = root / config["paths"]["pdf_path"]
    ordered_figures = [
        fig_paths["01_dataset_composition"],
        fig_paths["02_temporal_distribution"],
        fig_paths["03_usages_per_target_word"],
        fig_paths["04_context_length"],
        fig_paths["05_lexical_characteristics"],
        fig_paths["06_embedding_structure"],
        fig_paths["07_temporal_embedding_similarity"],
        fig_paths["08_definition_usage_similarity"],
    ]
    build_eda_pdf(ordered_figures, pdf_path)

    return {
        "root": str(root),
        "usage_records": int(len(usage_df)),
        "unique_target_words": int(usage_df["word"].nunique(dropna=True)),
        "periods": int(usage_df["period_label"].nunique(dropna=True)),
        "figures": ordered_figures,
        "tables": list(tables.values()),
        "pdf": str(pdf_path),
        "findings": str(findings_path),
        "stats": stats,
    }


if __name__ == "__main__":
    summary = run_eda()
    print(
        "EDA complete. "
        f"records={summary['usage_records']}, words={summary['unique_target_words']}, periods={summary['periods']}, "
        f"pdf={summary['pdf']}"
    )
