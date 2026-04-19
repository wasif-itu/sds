from __future__ import annotations

import json
from pathlib import Path

from scripts.spatial_analysis_utils import run_full_analysis


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    results = run_full_analysis(PROJECT_ROOT, save_results=True)
    gdf = results["gdf"]
    global_moran = results["global_moran"]
    regression = results["regression"]
    sensitivity = results["sensitivity"]
    saved_paths = results["saved_paths"]

    assert len(gdf) > 50, f"Expected more than 50 tehsils, found {len(gdf)}"
    assert "spi" in gdf.columns
    assert gdf["spi"].notna().all()
    assert {"HH", "HL", "LH", "LL", "Not Significant"}.intersection(set(gdf["spi_cluster"].unique()))
    assert global_moran["moran_i"].between(-1, 1).all()
    assert global_moran["p_value"].between(0, 1).all()
    assert regression["selected_model"] in {"spatial_lag", "spatial_error"}
    assert -1 <= regression["residual_moran"]["moran_i"] <= 1
    assert len(sensitivity) >= 5

    for name, path in saved_paths.items():
        assert path.exists(), f"Missing output for {name}: {path}"

    summary = {
        "rows": int(len(gdf)),
        "priority_count": int(gdf["priority_high_scenic_low_access"].sum()),
        "selected_model": regression["selected_model"],
        "spi_source": gdf["spi_source"].iloc[0],
        "global_moran": global_moran.to_dict(orient="records"),
        "top_priority_tehsils": gdf.loc[gdf["priority_high_scenic_low_access"], "shapeName"].head(10).tolist(),
        "verified_outputs": {name: str(path) for name, path in saved_paths.items()},
    }

    out = PROJECT_ROOT / "outputs" / "spatial_disparity_analysis" / "verification_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Verification completed successfully.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
