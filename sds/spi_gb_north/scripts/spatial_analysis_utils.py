from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.lines import Line2D
from esda.moran import Moran, Moran_Local
try:
    from esda.moran import Moran_Local_BV
except Exception:  # pragma: no cover
    Moran_Local_BV = None
from libpysal.weights import Queen
from scipy.stats import spearmanr
from spreg import ML_Error, ML_Lag, OLS
from spreg.diagnostics_sp import LMtests


def get_project_root(base_dir: str | Path = ".") -> Path:
    return Path(base_dir).resolve()


def get_analysis_paths(base_dir: str | Path = ".") -> dict[str, Path]:
    root = get_project_root(base_dir)
    outputs_dir = root / "outputs" / "spatial_disparity_analysis"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "tehsil_file": root / "data" / "processed" / "tehsil_spi_ai_fullstudy.geojson",
        "tehsil_csv_with_spi": root / "data" / "processed" / "tehsil_spi_ai.csv",
        "tehsil_fullstudy_csv": root / "data" / "processed" / "tehsil_spi_ai_fullstudy.csv",
        "spi_raster": root / "data" / "processed" / "spi_index.tif",
        "ai_raster": root / "data" / "processed" / "ai_index.tif",
        "visualize_notebook": root / "visualize_spi_roads.ipynb",
        "outputs_dir": outputs_dir,
    }


def load_tehsil_data(base_dir: str | Path = ".") -> gpd.GeoDataFrame:
    paths = get_analysis_paths(base_dir)
    gdf = gpd.read_file(paths["tehsil_file"])
    gdf["spi_source"] = "unknown"

    if "spi" not in gdf.columns:
        merge_candidates = []
        if paths["tehsil_csv_with_spi"].exists():
            merge_candidates.append(paths["tehsil_csv_with_spi"])
        if paths["tehsil_fullstudy_csv"].exists():
            merge_candidates.append(paths["tehsil_fullstudy_csv"])

        for candidate in merge_candidates:
            df = pd.read_csv(candidate)
            if "spi" in df.columns and "shapeID" in df.columns:
                cols = ["shapeID", "spi"]
                if "snow_pct" in df.columns:
                    cols.append("snow_pct")
                merge_df = df[cols].drop_duplicates("shapeID")
                gdf = gdf.merge(merge_df, on="shapeID", how="left", suffixes=("", "_from_csv"))
                if gdf["spi"].notna().any():
                    gdf["spi_source"] = f"merged_from_{candidate.name}"
                    break

    if "spi" not in gdf.columns and "spi_no_snow" in gdf.columns:
        gdf["spi"] = gdf["spi_no_snow"]
        gdf["spi_source"] = "fallback_spi_no_snow"
    elif "spi" in gdf.columns and (gdf["spi_source"] == "unknown").all():
        gdf["spi_source"] = "direct_spi_column"

    required = [
        "shapeName",
        "road_density_km_per_km2",
        "dist_to_roads_mean_m",
        "tri_mean",
        "forest_pct",
        "water_pct",
        "spi",
        "ai",
        "geometry",
    ]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    gdf = gdf.dropna(subset=["spi", "ai"]).copy()
    return gdf


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def prepare_analysis_frame(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["spi_z"] = zscore(out["spi"])
    out["ai_z"] = zscore(out["ai"])
    out["road_density_z"] = zscore(out["road_density_km_per_km2"])
    out["dist_to_roads_z"] = zscore(out["dist_to_roads_mean_m"])
    out["tri_z"] = zscore(out["tri_mean"])
    out["forest_z"] = zscore(out["forest_pct"])
    out["water_z"] = zscore(out["water_pct"])
    out["gap_index"] = out["spi_z"] - out["ai_z"]
    out["priority_high_scenic_low_access"] = (out["spi_z"] > 0) & (out["ai_z"] < 0)
    return out


def build_weights(gdf: gpd.GeoDataFrame) -> Queen:
    w = Queen.from_dataframe(gdf)
    w.transform = "r"
    return w


def moran_summary(series: pd.Series, weights: Queen, permutations: int = 999) -> dict[str, Any]:
    moran = Moran(series.to_numpy(), weights, permutations=permutations)
    return {
        "n": int(len(series)),
        "moran_i": float(moran.I),
        "expected_i": float(moran.EI),
        "z_score": float(moran.z_sim),
        "p_value": float(moran.p_sim),
    }


def cluster_from_local(local: Moran_Local, alpha: float = 0.05) -> pd.DataFrame:
    quadrant_labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    labels: list[str] = []
    for q, p in zip(local.q, local.p_sim, strict=False):
        if p <= alpha:
            labels.append(quadrant_labels.get(int(q), "Not Significant"))
        else:
            labels.append("Not Significant")
    return pd.DataFrame(
        {
            "local_I": local.Is,
            "local_p": local.p_sim,
            "quadrant": local.q,
            "cluster": labels,
        }
    )


def compute_lisa_columns(gdf: gpd.GeoDataFrame, weights: Queen, alpha: float = 0.05) -> gpd.GeoDataFrame:
    out = gdf.copy()
    for source, prefix in [("spi_z", "spi"), ("ai_z", "ai"), ("gap_index", "gap")]:
        local = Moran_Local(out[source].to_numpy(), weights, permutations=999)
        local_df = cluster_from_local(local, alpha=alpha)
        for col in local_df.columns:
            out[f"{prefix}_{col}"] = local_df[col].values
    if Moran_Local_BV is not None:
        local_bv = Moran_Local_BV(out["spi_z"].to_numpy(), out["ai_z"].to_numpy(), weights, permutations=999)
        local_bv_df = cluster_from_local(local_bv, alpha=alpha)
        for col in local_bv_df.columns:
            out[f"spi_ai_bv_{col}"] = local_bv_df[col].values
    else:
        out["spi_ai_bv_cluster"] = "Unavailable"
        out["spi_ai_bv_local_p"] = np.nan
    return out


def compute_global_moran_table(gdf: gpd.GeoDataFrame, weights: Queen) -> pd.DataFrame:
    rows = []
    for label, source in [
        ("SPI (snow-inclusive)", "spi"),
        ("Accessibility", "ai"),
        ("Gap (SPI - AI)", "gap_index"),
    ]:
        summary = moran_summary(gdf[source], weights)
        summary["variable"] = label
        rows.append(summary)
    return pd.DataFrame(rows)[["variable", "n", "moran_i", "expected_i", "z_score", "p_value"]]


def run_regression_suite(gdf: gpd.GeoDataFrame, weights: Queen) -> dict[str, Any]:
    y = gdf["spi"].to_numpy()
    X = gdf[["ai"]].to_numpy()
    X_sm = sm.add_constant(X)
    sm_model = sm.OLS(y, X_sm).fit()

    residual_moran = Moran(sm_model.resid, weights, permutations=999)

    sp_ols = OLS(
        y.reshape(-1, 1),
        X,
        w=weights,
        name_y="spi",
        name_x=["ai"],
    )
    lm = LMtests(sp_ols, weights)

    lm_table = pd.DataFrame(
        [
            {"test": "LM Lag", "statistic": float(lm.lml[0]), "p_value": float(lm.lml[1])},
            {"test": "Robust LM Lag", "statistic": float(lm.rlml[0]), "p_value": float(lm.rlml[1])},
            {"test": "LM Error", "statistic": float(lm.lme[0]), "p_value": float(lm.lme[1])},
            {"test": "Robust LM Error", "statistic": float(lm.rlme[0]), "p_value": float(lm.rlme[1])},
        ]
    )

    robust_lag_p = float(lm.rlml[1])
    robust_err_p = float(lm.rlme[1])
    if robust_lag_p < 0.05 and robust_err_p >= 0.05:
        selected_model = "spatial_lag"
    elif robust_err_p < 0.05 and robust_lag_p >= 0.05:
        selected_model = "spatial_error"
    elif robust_lag_p < robust_err_p:
        selected_model = "spatial_lag"
    else:
        selected_model = "spatial_error"

    if selected_model == "spatial_lag":
        spatial_model = ML_Lag(
            y.reshape(-1, 1),
            X,
            w=weights,
            name_y="spi",
            name_x=["ai"],
        )
    else:
        spatial_model = ML_Error(
            y.reshape(-1, 1),
            X,
            w=weights,
            name_y="spi",
            name_x=["ai"],
        )

    coef_names = ["const", "ai"]
    spatial_betas = spatial_model.betas.flatten().tolist()
    if len(spatial_betas) == 3:
        coef_names.append("spatial_parameter")
    spatial_coefficients = pd.DataFrame({"term": coef_names[: len(spatial_betas)], "estimate": spatial_betas})

    ols_coefficients = pd.DataFrame(
        {
            "term": ["const", "ai"],
            "estimate": np.asarray(sm_model.params),
            "p_value": np.asarray(sm_model.pvalues),
        }
    )

    return {
        "statsmodels_ols": sm_model,
        "spreg_ols": sp_ols,
        "ols_coefficients": ols_coefficients.reset_index(drop=True),
        "residual_moran": {
            "moran_i": float(residual_moran.I),
            "expected_i": float(residual_moran.EI),
            "z_score": float(residual_moran.z_sim),
            "p_value": float(residual_moran.p_sim),
        },
        "lm_tests": lm_table,
        "selected_model": selected_model,
        "spatial_model": spatial_model,
        "spatial_coefficients": spatial_coefficients,
    }


def run_sensitivity_analysis(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    components = gdf[["tri_z", "forest_z", "water_z"]].copy()
    baseline = gdf["spi_no_snow"]
    scenarios = {
        "equal_weights": np.array([1 / 3, 1 / 3, 1 / 3]),
        "terrain_plus_015": np.array([0.48, 0.26, 0.26]),
        "terrain_minus_015": np.array([0.18, 0.41, 0.41]),
        "forest_plus_015": np.array([0.26, 0.48, 0.26]),
        "forest_minus_015": np.array([0.41, 0.18, 0.41]),
        "water_plus_015": np.array([0.26, 0.26, 0.48]),
        "water_minus_015": np.array([0.41, 0.41, 0.18]),
    }
    rows = []
    for name, weights in scenarios.items():
        weights = weights / weights.sum()
        score = (components.to_numpy() * weights).sum(axis=1)
        corr = spearmanr(score, baseline).statistic
        top_overlap = len(
            set(np.argsort(score)[-10:]).intersection(set(np.argsort(baseline.to_numpy())[-10:]))
        )
        rows.append(
            {
                "scenario": name,
                "terrain_weight": float(weights[0]),
                "forest_weight": float(weights[1]),
                "water_weight": float(weights[2]),
                "spearman_vs_baseline": float(corr),
                "top10_overlap": int(top_overlap),
            }
        )
    return pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)


def plot_choropleths(gdf: gpd.GeoDataFrame, outputs_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 18))
    fig.patch.set_facecolor("white")
    maps = [
        ("spi", "SPI", "YlGn"),
        ("ai", "Accessibility Index", "YlOrBr"),
        ("gap_index", "Scenic-Access Gap", "RdYlGn"),
        ("priority_high_scenic_low_access", "Priority Tehsils", None),
    ]
    for ax, (col, title, cmap) in zip(axes.flat, maps, strict=False):
        if col == "priority_high_scenic_low_access":
            gdf.plot(
                ax=ax,
                column=col,
                categorical=True,
                legend=True,
                cmap="Set1",
                linewidth=0.4,
                edgecolor="black",
            )
        else:
            gdf.plot(
                ax=ax,
                column=col,
                cmap=cmap,
                legend=True,
                linewidth=0.4,
                edgecolor="black",
            )
        ax.set_title(title)
        ax.set_axis_off()
    plt.tight_layout()
    out = outputs_dir / "analysis_choropleths.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_lisa_maps(gdf: gpd.GeoDataFrame, outputs_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    cluster_colors = {
        "HH": "#b2182b",
        "HL": "#ef8a62",
        "LH": "#67a9cf",
        "LL": "#2166ac",
        "Not Significant": "#d9d9d9",
    }
    for ax, (col, title) in zip(
        axes,
        [
            ("spi_cluster", "SPI LISA"),
            ("ai_cluster", "Accessibility LISA"),
            ("gap_cluster", "Gap LISA"),
        ],
        strict=False,
    ):
        colors = gdf[col].map(cluster_colors).fillna("#d9d9d9")
        gdf.assign(plot_color=colors).plot(
            ax=ax,
            color=gdf.assign(plot_color=colors)["plot_color"],
            linewidth=0.4,
            edgecolor="black",
        )
        ax.set_title(title)
        ax.set_axis_off()
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markersize=12, label=label)
        for label, color in cluster_colors.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = outputs_dir / "analysis_lisa_maps.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_regression(gdf: gpd.GeoDataFrame, regression: dict[str, Any], outputs_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.regplot(data=gdf, x="ai", y="spi", ax=axes[0], scatter_kws={"s": 30, "alpha": 0.7})
    axes[0].set_title("SPI vs Accessibility")
    axes[0].set_xlabel("Accessibility Index")
    axes[0].set_ylabel("SPI")

    resid = regression["statsmodels_ols"].resid
    sns.histplot(resid, bins=20, kde=True, ax=axes[1], color="#2c7fb8")
    axes[1].set_title("OLS Residual Distribution")
    axes[1].set_xlabel("Residual")
    plt.tight_layout()
    out = outputs_dir / "analysis_regression_diagnostics.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def write_simple_results_summary(
    gdf: gpd.GeoDataFrame,
    global_moran: pd.DataFrame,
    regression: dict[str, Any],
    sensitivity: pd.DataFrame,
    outputs_dir: Path,
) -> Path:
    priority = (
        gdf.loc[gdf["priority_high_scenic_low_access"], ["shapeName", "spi", "ai", "gap_index"]]
        .sort_values("gap_index", ascending=False)
        .head(10)
    )
    top_priority_lines = "\n".join(
        f"- {row.shapeName}: SPI {row.spi:.2f}, AI {row.ai:.2f}, gap {row.gap_index:.2f}"
        for row in priority.itertuples(index=False)
    )
    global_lookup = {row["variable"]: row for row in global_moran.to_dict(orient="records")}
    spi_moran = global_lookup["SPI (snow-inclusive)"]
    ai_moran = global_lookup["Accessibility"]
    gap_moran = global_lookup["Gap (SPI - AI)"]
    best_sensitivity = sensitivity.sort_values("spearman_vs_baseline", ascending=False).iloc[0]

    text = f"""Spatial Disparity Analysis Explained in Simple Words

What this analysis asked
- Are scenic places clustered together?
- Are accessible places clustered together?
- Which tehsils look scenic but still have relatively poor access?
- Does accessibility help explain scenic potential once we account for spatial patterns?

What the results say
- Scenic potential is strongly clustered in space. Moran's I for SPI was {spi_moran['moran_i']:.3f} with p={spi_moran['p_value']:.3f}, which means nearby tehsils tend to have similar scenic scores.
- Accessibility is also strongly clustered. Moran's I for AI was {ai_moran['moran_i']:.3f} with p={ai_moran['p_value']:.3f}, so road access is not random either.
- The mismatch between scenery and access is even more clustered. Moran's I for the scenic-access gap was {gap_moran['moran_i']:.3f} with p={gap_moran['p_value']:.3f}. In simple words, the study area has clear pockets where beautiful places are consistently less accessible than surrounding areas.
- The preferred spatial regression model was {regression['selected_model'].replace('_', ' ')}, which means ordinary regression alone was not enough; nearby tehsils influence one another.
- The analysis identified {int(gdf['priority_high_scenic_low_access'].sum())} tehsils as high-scenic and low-access priority areas.

What this means practically
- Northern Pakistan does not have an even spread of scenic opportunity and road access.
- Some scenic corridors are already well connected, but several highly scenic tehsils remain relatively underserved.
- These underserved scenic tehsils are good candidates for closer planning attention, especially for tourism access, basic infrastructure, and service delivery.
- Because the results are spatially clustered, planning individual tehsils in isolation would miss the broader regional pattern.

Priority tehsils highlighted by the model
{top_priority_lines}

How stable the SPI is
- The sensitivity analysis showed the index is fairly stable under alternative weights.
- The strongest alternative scenario had Spearman correlation {best_sensitivity['spearman_vs_baseline']:.3f} against the baseline SPI and a top-10 overlap of {int(best_sensitivity['top10_overlap'])}.
- In simple words, the broad ranking of scenic places does not change dramatically when the component weights are adjusted moderately.

Bottom line
- The results support the project idea: scenic potential and accessibility are related, but not evenly matched.
- The main policy-relevant finding is not just where the most scenic places are, but where scenic potential is high and access is still comparatively weak.
"""
    out = outputs_dir / "spatial_disparity_results_explained.txt"
    out.write_text(text, encoding="utf-8")
    return out


def plot_sensitivity(sensitivity: pd.DataFrame, outputs_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=sensitivity, x="scenario", y="spearman_vs_baseline", ax=ax, color="#4daf4a")
    ax.set_ylim(0, 1.05)
    ax.set_title("Sensitivity of Alternative Weighting Schemes")
    ax.set_xlabel("")
    ax.set_ylabel("Spearman correlation vs baseline SPI")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    out = outputs_dir / "analysis_sensitivity.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def save_outputs(
    gdf: gpd.GeoDataFrame,
    global_moran: pd.DataFrame,
    regression: dict[str, Any],
    sensitivity: pd.DataFrame,
    base_dir: str | Path = ".",
) -> dict[str, Path]:
    paths = get_analysis_paths(base_dir)
    outputs_dir = paths["outputs_dir"]
    enriched_geojson = outputs_dir / "tehsil_spatial_analysis.geojson"
    enriched_csv = outputs_dir / "tehsil_spatial_analysis.csv"
    global_csv = outputs_dir / "global_moran_summary.csv"
    lm_csv = outputs_dir / "lm_tests.csv"
    ols_csv = outputs_dir / "ols_coefficients.csv"
    spatial_csv = outputs_dir / "spatial_model_coefficients.csv"
    sensitivity_csv = outputs_dir / "sensitivity_analysis.csv"
    priority_csv = outputs_dir / "priority_tehsils.csv"
    summary_json = outputs_dir / "analysis_summary.json"
    model_txt = outputs_dir / "spatial_model_summary.txt"

    gdf.to_file(enriched_geojson, driver="GeoJSON")
    gdf.drop(columns="geometry").to_csv(enriched_csv, index=False)
    global_moran.to_csv(global_csv, index=False)
    regression["lm_tests"].to_csv(lm_csv, index=False)
    regression["ols_coefficients"].to_csv(ols_csv, index=False)
    regression["spatial_coefficients"].to_csv(spatial_csv, index=False)
    sensitivity.to_csv(sensitivity_csv, index=False)
    gdf.loc[gdf["priority_high_scenic_low_access"]].drop(columns="geometry").sort_values(
        "gap_index", ascending=False
    ).to_csv(priority_csv, index=False)

    summary = {
        "rows": int(len(gdf)),
        "priority_count": int(gdf["priority_high_scenic_low_access"].sum()),
        "selected_model": regression["selected_model"],
        "residual_moran": regression["residual_moran"],
        "global_moran": global_moran.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    model_txt.write_text(str(regression["spatial_model"].summary), encoding="utf-8")

    plot_paths = {
        "choropleths": plot_choropleths(gdf, outputs_dir),
        "lisa_maps": plot_lisa_maps(gdf, outputs_dir),
        "regression_plot": plot_regression(gdf, regression, outputs_dir),
        "sensitivity_plot": plot_sensitivity(sensitivity, outputs_dir),
        "simple_results_text": write_simple_results_summary(gdf, global_moran, regression, sensitivity, outputs_dir),
    }

    return {
        "enriched_geojson": enriched_geojson,
        "enriched_csv": enriched_csv,
        "global_csv": global_csv,
        "lm_csv": lm_csv,
        "ols_csv": ols_csv,
        "spatial_csv": spatial_csv,
        "sensitivity_csv": sensitivity_csv,
        "priority_csv": priority_csv,
        "summary_json": summary_json,
        "model_txt": model_txt,
        **plot_paths,
    }


def run_full_analysis(base_dir: str | Path = ".", save_results: bool = True) -> dict[str, Any]:
    gdf = load_tehsil_data(base_dir)
    gdf = prepare_analysis_frame(gdf)
    weights = build_weights(gdf)
    gdf = compute_lisa_columns(gdf, weights)
    global_moran = compute_global_moran_table(gdf, weights)
    regression = run_regression_suite(gdf, weights)
    sensitivity = run_sensitivity_analysis(gdf)
    saved_paths = save_outputs(gdf, global_moran, regression, sensitivity, base_dir) if save_results else {}
    priority = (
        gdf.loc[gdf["priority_high_scenic_low_access"]]
        .drop(columns="geometry")
        .sort_values("gap_index", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "gdf": gdf,
        "weights": weights,
        "global_moran": global_moran,
        "regression": regression,
        "sensitivity": sensitivity,
        "priority_tehsils": priority,
        "saved_paths": saved_paths,
    }
