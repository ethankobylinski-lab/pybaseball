"""Streamlit application exposing MLB win probability insights."""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import polars as pl
import plotly.express as px
import streamlit as st

from app.data import WarehouseRepository
from features.pipeline import FeatureCollection, build_features
from features.registry import default_registry
from models.training import ModelTrainer, TrainingConfig

st.set_page_config(page_title="MLB Win Factors Lab", layout="wide")


def _win_rate(joined: pl.DataFrame) -> float:
    total = joined.height
    if total == 0:
        return float("nan")
    wins = joined.filter(pl.col("team_result") == "win").height
    return wins / total


def _format_percent(value: float) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.1%}"


def _with_flag_cast(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """Return a frame with the flag column coerced to integer markers."""

    return frame.with_columns(pl.col(column).cast(pl.Int8, strict=False).alias(column))


def _flagged_win_rate(team_games: pl.DataFrame, feature_frame: pl.DataFrame, flag_column: str) -> Dict[str, float]:
    frame = _with_flag_cast(feature_frame, flag_column)
    joined = team_games.join(frame, on=["game_pk", "team_id"], how="left")
    flagged = joined.filter(pl.col(flag_column) == 1)
    if flagged.height == 0:
        return {"win_rate": float("nan"), "baseline": _win_rate(joined), "n": 0}
    return {
        "win_rate": _win_rate(flagged),
        "baseline": _win_rate(joined),
        "n": flagged.height,
    }


def _mean_or_nan(frame: pl.DataFrame, column: str) -> float:
    if frame.height == 0:
        return float("nan")
    value = frame.select(pl.col(column).mean()).item(0, 0)
    return float(value) if value is not None else float("nan")


def _shutdown_summary(team_games: pl.DataFrame, shutdown_frame: pl.DataFrame) -> Dict[str, float]:
    joined = team_games.join(shutdown_frame, on=["game_pk", "team_id"], how="left")
    winners = joined.filter(pl.col("team_result") == "win")
    losers = joined.filter(pl.col("team_result") == "loss")
    return {
        "winners": _mean_or_nan(winners, "shutdown_inning_after_scoring"),
        "losers": _mean_or_nan(losers, "shutdown_inning_after_scoring"),
    }


@dataclass(slots=True)
class FactorOption:
    """Metadata describing a binary win-factor feature for UI storytelling."""

    key: str
    label: str
    frame: pl.DataFrame
    column: str
    description: str


def _win_rate_lift_summary(team_games: pl.DataFrame, factors: Sequence[FactorOption]) -> pl.DataFrame:
    rows: List[Dict[str, Any]] = []
    for option in factors:
        frame = _with_flag_cast(option.frame, option.column)
        joined = team_games.join(frame, on=["game_pk", "team_id"], how="left")
        flagged = joined.filter(pl.col(option.column) == 1)
        baseline = _win_rate(joined)
        win_rate = _win_rate(flagged)
        lift = (
            win_rate - baseline
            if not math.isnan(win_rate) and not math.isnan(baseline)
            else float("nan")
        )
        rows.append(
            {
                "Factor": option.label,
                "Win Rate": win_rate,
                "Baseline": baseline,
                "Lift": lift,
                "Games": flagged.height,
                "Description": option.description,
                "Key": option.key,
            }
        )
    return pl.DataFrame(rows)


def _factor_trend(
    team_games: pl.DataFrame, feature_frame: pl.DataFrame, column: str
) -> pl.DataFrame:
    frame = _with_flag_cast(feature_frame, column)
    joined = team_games.join(frame, on=["game_pk", "team_id"], how="left")
    if "season" not in joined.columns:
        return pl.DataFrame({"season": [], "Win Rate": [], "Baseline": [], "Games": []})
    joined = joined.with_columns(
        pl.when(pl.col("team_result") == "win").then(1).otherwise(0).alias("win_flag")
    )
    summary = (
        joined.group_by("season")
        .agg(
            pl.col("win_flag").mean().alias("Baseline"),
            pl.when(pl.col(column) == 1)
            .then(pl.col("win_flag"))
            .otherwise(None)
            .mean()
            .alias("Win Rate"),
            pl.when(pl.col(column) == 1).then(1).otherwise(0).sum().alias("Games"),
        )
        .sort("season")
    )
    return summary


def _factor_matrix(
    team_games: pl.DataFrame,
    row_feature: FactorOption,
    col_feature: FactorOption,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    row_frame = _with_flag_cast(row_feature.frame, row_feature.column)
    col_frame = _with_flag_cast(col_feature.frame, col_feature.column)
    joined = (
        team_games.join(row_frame, on=["game_pk", "team_id"], how="left")
        .join(col_frame, on=["game_pk", "team_id"], how="left")
        .with_columns(
            pl.when(pl.col("team_result") == "win").then(1).otherwise(0).alias("win_flag")
        )
    )

    summary = (
        joined.group_by([row_feature.column, col_feature.column])
        .agg(
            pl.col("win_flag").mean().alias("win_rate"),
            pl.len().alias("games"),
        )
        .sort([row_feature.column, col_feature.column])
    )

    matrix = (
        summary.pivot(index=row_feature.column, columns=col_feature.column, values="win_rate")
        .sort(row_feature.column)
    )
    counts = (
        summary.pivot(index=row_feature.column, columns=col_feature.column, values="games")
        .sort(row_feature.column)
    )
    return matrix, counts


def _collect_factor_options(collection: FeatureCollection) -> List[FactorOption]:
    metadata = collection.metadata
    options: Dict[Tuple[str, str], FactorOption] = {}

    manual = [
        ("Score First", "scored_first", "scored_first"),
        ("Big Inning", "big_inning", "big_inning"),
        ("Answered Back", "answered_back", "answered_back_next_half"),
        ("Shutdown Inning", "shutdown_after", "shutdown_inning_after_scoring"),
    ]

    for label, key, column in manual:
        if key in collection and column in collection[key].columns:
            description = metadata.get(key, label)
            options[(key, column)] = FactorOption(
                key=key,
                label=label,
                frame=collection[key],
                column=column,
                description=description,
            )

    for key in collection:
        if key == "team_games":
            continue
        frame = collection[key]
        candidate_cols = [col for col in frame.columns if col not in {"game_pk", "team_id"}]
        if not candidate_cols:
            continue
        for column in candidate_cols:
            unique = (
                frame.select(pl.col(column).cast(pl.Int8, strict=False).drop_nulls().unique())
                .to_series()
                .to_list()
            )
            values = {val for val in unique if val is not None}
            if not values:
                continue
            if values.issubset({0, 1}):
                description = metadata.get(key, key)
                label = description
                if len(candidate_cols) > 1 and column != key:
                    label = f"{description} ({column})"
                options.setdefault(
                    (key, column),
                    FactorOption(
                        key=key,
                        label=label,
                        frame=frame,
                        column=column,
                        description=description,
                    ),
                )

    return list(options.values())


def _format_delta(value: float) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1%}"


def _combo_win_rate_summary(
    team_games: pl.DataFrame,
    factors: Sequence[FactorOption],
    max_size: int = 3,
    min_games: int = 25,
) -> pl.DataFrame:
    baseline = _win_rate(team_games)
    rows: List[Dict[str, Any]] = []
    usable = [option for option in factors if option.frame.height > 0]
    if len(usable) < 2:
        return pl.DataFrame({"Combo": [], "Win Rate": [], "Lift": [], "Games": []})

    max_k = min(max_size, len(usable))
    for size in range(2, max_k + 1):
        for combo in combinations(usable, size):
            subset = team_games
            for option in combo:
                frame = _with_flag_cast(option.frame, option.column)
                subset = subset.join(frame, on=["game_pk", "team_id"], how="left")
                subset = subset.filter(pl.col(option.column) == 1)
            games = subset.height
            if games < min_games:
                continue
            win_rate = _win_rate(subset)
            lift = (
                win_rate - baseline
                if not math.isnan(win_rate) and not math.isnan(baseline)
                else float("nan")
            )
            rows.append(
                {
                    "Combo": " + ".join(option.label for option in combo),
                    "Win Rate": win_rate,
                    "Lift": lift,
                    "Games": games,
                }
            )

    if not rows:
        return pl.DataFrame({"Combo": [], "Win Rate": [], "Lift": [], "Games": []})

    return pl.DataFrame(rows).sort("Lift", descending=True)


def _assemble_dataset(collection: FeatureCollection, selected_features: Iterable[str]) -> tuple[pl.DataFrame, List[str]]:
    dataset = collection.team_games
    feature_columns: List[str] = []
    for key in selected_features:
        frame = collection[key]
        non_key_cols = [col for col in frame.columns if col not in {"game_pk", "team_id"}]
        dataset = dataset.join(frame, on=["game_pk", "team_id"], how="left")
        feature_columns.extend(non_key_cols)

    dataset = dataset.with_columns(
        pl.when(pl.col("team_result") == "win").then(1).otherwise(0).alias("team_win")
    )

    if feature_columns:
        dataset = dataset.with_columns([pl.col(col).fill_null(0).cast(pl.Float64) for col in feature_columns])

    return dataset, feature_columns


def render_coach_briefing(collection: FeatureCollection) -> None:
    st.header("Coach & Player Briefing")
    st.caption(
        "Quick-hitting talking points you can share in the clubhouse or classroom. "
        "All win rates respect the seasons you selected on the left."
    )

    team_games = collection.team_games
    if team_games.is_empty():
        st.info("Load team-game data to see the briefing.")
        return

    factors = _collect_factor_options(collection)
    if not factors:
        st.info("Add binary game-flow features in the sidebar to populate the briefing.")
        return

    summary = _win_rate_lift_summary(team_games, factors)
    if summary.is_empty():
        st.info("No qualifying games for the selected factors.")
        return

    top_cards = summary.sort("Lift", descending=True).head(min(3, summary.height))
    columns = st.columns(len(top_cards))
    for idx, row in enumerate(top_cards.iter_rows(named=True)):
        delta = _format_delta(row["Lift"])
        columns[idx].metric(
            row["Factor"],
            _format_percent(row["Win Rate"]),
            delta=delta,
            help=row.get("Description", row["Factor"]),
        )
        columns[idx].caption(f"Games: {row['Games']:,}")

    st.markdown("---")
    st.subheader("Top Insights to Share")

    bullets = []
    for row in top_cards.iter_rows(named=True):
        baseline = _format_percent(row["Baseline"])
        lift = _format_delta(row["Lift"])
        bullets.append(
            f"- **{row['Factor']}** wins {_format_percent(row['Win Rate'])} of the time "
            f"({lift} vs baseline {baseline}) across {row['Games']:,} games."
        )
    st.markdown("\n".join(bullets))

    fig = px.bar(
        summary.sort("Lift", descending=True).to_pandas(),
        x="Factor",
        y="Lift",
        text="Lift",
        hover_data={"Win Rate": ":.1%", "Baseline": ":.1%", "Games": True},
        labels={"Lift": "Win Rate Lift"},
        color="Lift",
        color_continuous_scale="Tealgrn",
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(yaxis_tickformat=".0%", uniformtext_minsize=10, uniformtext_mode="hide")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Winning Recipe Combos")
    st.caption(
        "Use these combinations during practice planning to spotlight what happens when multiple factors line up."
    )
    combo_df = _combo_win_rate_summary(team_games, factors)
    if combo_df.is_empty():
        st.info("Need more games per combination before combo insights become reliable.")
    else:
        combo_pd = combo_df.head(10).to_pandas()
        combo_fig = px.bar(
            combo_pd,
            x="Combo",
            y="Lift",
            text="Win Rate",
            hover_data={"Win Rate": ":.1%", "Games": True},
            labels={"Lift": "Win Rate Lift"},
            color="Lift",
            color_continuous_scale="Bluered_r",
        )
        combo_fig.update_traces(texttemplate="Win: %{text:.1%}", textposition="outside")
        combo_fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(combo_fig, use_container_width=True)
        st.dataframe(combo_pd, use_container_width=True)

    with st.expander("How to present this to players"):
        st.markdown(
            "Focus on one or two headline factors, show the lift chart for visuals, "
            "then walk through the winning recipe combos as actionable goals for the series or practice block."
        )


def render_discoveries(collection: FeatureCollection) -> None:
    st.header("Discoveries")
    team_games = collection.team_games
    st.caption("Explore observed win rates for the selected seasons and features.")

    factors = _collect_factor_options(collection)
    factor_by_key = {factor.key: factor for factor in factors}

    cols = st.columns(3)
    manual_spotlights = [
        ("scored_first", "Score First"),
        ("big_inning", "Big Inning"),
        ("answered_back", "Answered Back"),
    ]
    for idx, (key, label) in enumerate(manual_spotlights):
        if key not in factor_by_key or idx >= len(cols):
            continue
        option = factor_by_key[key]
        stats = _flagged_win_rate(team_games, option.frame, option.column)
        cols[idx].metric(label, _format_percent(stats["win_rate"]), help=option.description)
        cols[idx].write(f"Baseline: {_format_percent(stats['baseline'])} (n={stats['n']})")

    if "shutdown_after" in factor_by_key:
        st.subheader("Shutdown Inning Quality")
        option = factor_by_key["shutdown_after"]
        summary = _shutdown_summary(team_games, option.frame)
        st.write(
            f"Teams that eventually won posted a shutdown rate of {_format_percent(summary['winners'])} compared to {_format_percent(summary['losers'])} in losses."
        )

    if "two_out_runs" in collection:
        st.subheader("Two-Out Production")
        frame = collection.two_out_runs
        league_rate = _mean_or_nan(frame, "two_out_run_share")
        st.write(f"Across the selection, two-out runs accounted for {_format_percent(league_rate)} of total scoring.")
        st.dataframe(
            frame.select(["game_pk", "team_id", "two_out_runs", "two_out_run_share"]).head(50),
            use_container_width=True,
        )

    if factors:
        st.subheader("Win Impact Overview")
        lift_df = _win_rate_lift_summary(team_games, factors)
        if not lift_df.is_empty():
            fig = px.bar(
                lift_df.to_pandas(),
                x="Factor",
                y="Lift",
                text="Lift",
                hover_data={"Win Rate": ":.1%", "Baseline": ":.1%", "Games": True},
                labels={"Lift": "Win Rate Lift"},
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig.update_layout(yaxis_tickformat=".0%", uniformtext_minsize=10, uniformtext_mode="hide")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Seasonal Trend")
        options_by_label = {factor.label: factor for factor in factors}
        trend_factor = st.selectbox("Select a factor to trend", list(options_by_label.keys()))
        trend_option = options_by_label[trend_factor]
        trend_df = _factor_trend(team_games, trend_option.frame, trend_option.column)
        if trend_df.is_empty():
            st.info("Season column unavailable for trend chart.")
        else:
            trend_pd = trend_df.to_pandas()
            trend_fig = px.line(
                trend_pd,
                x="season",
                y=["Win Rate", "Baseline"],
                markers=True,
                labels={"value": "Win Rate", "season": "Season", "variable": "Metric"},
            )
            trend_fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(trend_fig, use_container_width=True)

        if len(factors) >= 2:
            st.markdown("---")
            st.subheader("Factor Interaction Matrix")
            option_labels = list(options_by_label.keys())
            row_label = st.selectbox("Row factor", option_labels, key="matrix_row")
            col_candidates = [label for label in option_labels if label != row_label]
            col_label = st.selectbox("Column factor", col_candidates, key="matrix_col")
            row_option = options_by_label[row_label]
            col_option = options_by_label[col_label]
            matrix, counts = _factor_matrix(team_games, row_option, col_option)
            if matrix.is_empty():
                st.info("Not enough data to compute interaction matrix for the selected factors.")
            else:
                matrix_pd = matrix.to_pandas().set_index(row_option.column)
                counts_pd = counts.to_pandas().set_index(row_option.column)

                def _format_flag(value: object) -> str:
                    if value == 1:
                        return "Yes"
                    if value == 0:
                        return "No"
                    return str(value)

                matrix_pd.index = [f"{row_label}: {_format_flag(idx)}" for idx in matrix_pd.index]
                matrix_pd.columns = [f"{col_label}: {_format_flag(col)}" for col in matrix_pd.columns]
                counts_pd.index = [f"{row_label}: {_format_flag(idx)}" for idx in counts_pd.index]
                counts_pd.columns = [f"{col_label}: {_format_flag(col)}" for col in counts_pd.columns]

                heatmap_fig = px.imshow(
                    matrix_pd,
                    text_auto=".1%",
                    color_continuous_scale="Blues",
                    labels={"color": "Win Rate"},
                )
                heatmap_fig.update_layout(yaxis_title=row_label, xaxis_title=col_label)
                st.plotly_chart(heatmap_fig, use_container_width=True)
                st.caption("Cell annotations show win rate; hover to view game counts.")
                st.dataframe(
                    counts_pd,
                    use_container_width=True,
                )

    st.subheader("What-if Explorer")

    if not factors:
        st.info("Add binary features to run combinations.")
        return

    selected = st.multiselect("Select factors to combine", list(options_by_label.keys()))
    if selected:
        dataset = team_games
        for label in selected:
            option = options_by_label[label]
            frame = _with_flag_cast(option.frame, option.column)
            dataset = dataset.join(frame, on=["game_pk", "team_id"], how="left")
            dataset = dataset.filter(pl.col(option.column) == 1)
        probability = _win_rate(dataset)
        st.success(f"Historical win rate with {', '.join(selected)}: {_format_percent(probability)} (n={dataset.height})")
    else:
        st.info("Choose factors above to evaluate historical win percentages.")


def render_optimization_lab(collection: FeatureCollection, selected_features: List[str]) -> None:
    st.header("Optimization Lab")
    st.caption(
        "Stack-rank factors, inspect relationships, and challenge assumptions like whether barrel rate or errors move wins."
    )

    dataset, feature_columns = _assemble_dataset(collection, selected_features)
    if dataset.is_empty() or not feature_columns:
        st.info("Materialise numeric features via the sidebar to run optimization studies.")
        return

    default_selection = feature_columns[: min(6, len(feature_columns))]
    analysis_features = st.multiselect(
        "Factors to compare",
        feature_columns,
        default=default_selection,
        help="Pick metrics to evaluate against win percentage. We'll compute correlations and visuals automatically.",
    )

    if not analysis_features:
        st.info("Select at least one factor to compare against win probability.")
        return

    corr_rows: List[Dict[str, Any]] = []
    for feature in analysis_features:
        try:
            corr = dataset.select(
                pl.corr(
                    pl.col("team_win").cast(pl.Float64),
                    pl.col(feature).cast(pl.Float64, strict=False),
                )
            ).item(0, 0)
        except Exception:  # pragma: no cover - defensive fallback
            corr = float("nan")
        corr_rows.append(
            {
                "Factor": feature,
                "Correlation": float(corr) if corr is not None else float("nan"),
            }
        )

    corr_df = pl.DataFrame(corr_rows)
    corr_pd = corr_df.sort("Correlation", descending=True, nulls_last=True).to_pandas()
    corr_fig = px.bar(
        corr_pd,
        x="Factor",
        y="Correlation",
        color="Correlation",
        color_continuous_scale="RdBu",
        labels={"Correlation": "Correlation with Win"},
    )
    corr_fig.update_layout(yaxis_tickformat=".2f")
    st.plotly_chart(corr_fig, use_container_width=True)
    st.dataframe(corr_pd, use_container_width=True)

    st.markdown("---")

    feature_choice = st.selectbox("Inspect distribution by result", analysis_features)
    distribution_df = (
        dataset.select([feature_choice, "team_result"])
        .drop_nulls()
        .limit(10000)
        .to_pandas()
    )
    if distribution_df.empty:
        st.info("Not enough data for the selected feature.")
    else:
        box_fig = px.box(
            distribution_df,
            x="team_result",
            y=feature_choice,
            labels={"team_result": "Result", feature_choice: feature_choice},
            points="suspectedoutliers",
        )
        st.plotly_chart(box_fig, use_container_width=True)

    if len(analysis_features) >= 2:
        st.markdown("---")
        st.subheader("Factor Interaction Heatmap")
        x_feature = st.selectbox("X-axis factor", analysis_features, key="opt_x")
        y_candidates = [feature for feature in analysis_features if feature != x_feature]
        if y_candidates:
            y_feature = st.selectbox("Y-axis factor", y_candidates, key="opt_y")
            interaction_df = (
                dataset.select([x_feature, y_feature, "team_win"])
                .drop_nulls()
                .with_columns(
                    pl.col(x_feature).round(3).alias(x_feature),
                    pl.col(y_feature).round(3).alias(y_feature),
                )
                .to_pandas()
            )
            if interaction_df.empty:
                st.info("No overlapping data for the selected pair of factors.")
            else:
                heatmap = px.density_heatmap(
                    interaction_df,
                    x=x_feature,
                    y=y_feature,
                    z="team_win",
                    histfunc="avg",
                    color_continuous_scale="Viridis",
                    labels={"z": "Avg Win Prob"},
                )
                heatmap.update_layout(coloraxis_colorbar=dict(title="Win%"))
                st.plotly_chart(heatmap, use_container_width=True)
                st.caption(
                    "Cells represent the average win probability where both factor values land in the same bucket."
                )


def render_model_builder(collection: FeatureCollection, selected_features: List[str]) -> None:
    st.header("Build Your Own Model")
    st.caption("Train interpretable models on the selected seasons without touching the pipeline code.")

    dataset, feature_columns = _assemble_dataset(collection, selected_features)
    if dataset.is_empty():
        st.warning("No records available. Load warehouse snapshots via data/warehouse.")
        return

    available_features = feature_columns
    if not available_features:
        st.info("No engineered features selected. Adjust the sidebar feature blocks to begin training.")
        return
    selected = st.multiselect("Model features", available_features, default=available_features[: min(5, len(available_features))])
    model_type = st.selectbox("Model", ["Logistic Regression", "LightGBM"], index=0)
    calibrate = st.checkbox("Calibrate probabilities", value=True, disabled=model_type == "LightGBM")

    st.dataframe(dataset.select(["game_pk", "team_id", "season", "team_result", *selected]).head(20), use_container_width=True)

    if st.button("Train Model", use_container_width=True):
        if not selected:
            st.warning("Select at least one feature to train a model.")
            return
        trainer = ModelTrainer(
            TrainingConfig(
                target_column="team_win",
                group_column="season",
                feature_columns=selected,
                calibration=calibrate,
            )
        )
        if model_type == "Logistic Regression":
            result = trainer.train_logistic(dataset.select(["team_win", "season", *selected]))
        else:
            result = trainer.train_lightgbm(dataset.select(["team_win", "season", *selected]))

        st.success(f"Trained {result.name} with AUC={result.auc:.3f}")
        st.write("Features:", ", ".join(result.feature_names))


def main() -> None:
    repo = WarehouseRepository()
    registry = default_registry()

    seasons_available = repo.available_seasons()
    with st.sidebar:
        st.header("Configuration")
        if not seasons_available:
            st.warning("Place parquet snapshots in data/warehouse to explore the app.")
        default_seasons = repo.default_selection() if seasons_available else []
        selected_seasons = st.multiselect(
            "Seasons / snapshots",
            seasons_available,
            default=default_seasons,
            help="Drop additional season parquet files into data/warehouse and they appear here automatically.",
        )
        include_all = st.checkbox("Materialise all registered features", value=False)
        definitions = registry.definitions()
        default_keys = [definition.key for definition in definitions if definition.default]
        feature_labels = {definition.key: f"{definition.key} · {definition.description}" for definition in definitions}
        selected_features = st.multiselect(
            "Feature blocks",
            options=default_keys,
            default=default_keys,
            format_func=lambda key: feature_labels.get(key, key),
            disabled=include_all,
        )

    try:
        tables = repo.load_tables(selected_seasons)
    except FileNotFoundError as err:
        st.error(str(err))
        return

    if tables.get("games") is None or tables["games"].is_empty():
        st.info("No games available for the selected seasons.")
        return

    include = ["*"] if include_all else selected_features or None
    collection = build_features(tables, include=include, registry=registry)
    if include_all:
        model_features = [key for key in collection if key != "team_games"]
    else:
        model_features = selected_features or [key for key in collection if key != "team_games"]

    tabs = st.tabs([
        "Coach & Player Briefing",
        "Discoveries",
        "Optimization Lab",
        "Model Builder",
    ])
    with tabs[0]:
        render_coach_briefing(collection)
    with tabs[1]:
        render_discoveries(collection)
    with tabs[2]:
        render_optimization_lab(collection, model_features)
    with tabs[3]:
        render_model_builder(collection, model_features)


if __name__ == "__main__":  # pragma: no cover - streamlit entrypoint
    main()
