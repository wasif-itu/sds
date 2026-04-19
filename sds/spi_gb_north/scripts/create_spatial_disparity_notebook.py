import json
from pathlib import Path


PROJECT_ROOT = Path(r"C:\Users\AtifA\Desktop\Wasif SDS Project\sds\sds\spi_gb_north")
NOTEBOOK_PATH = PROJECT_ROOT / "spatial_disparity_analysis.ipynb"


def lines(text: str) -> list[str]:
    return [line + "\n" for line in text.strip("\n").splitlines()]


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def main() -> None:
    nb = {
        "cells": [
            md_cell(
                """
# Spatial Disparity Analysis: SPI, Accessibility, LISA, and Spatial Regression

This notebook continues the project after preprocessing and tehsil-level SPI/AI creation.

It uses the latest processed tehsil file reflected in the current workflow:
- `data/processed/tehsil_spi_ai_fullstudy.geojson`

If that GeoJSON does not expose the final snow-inclusive `spi` field directly,
the helper code automatically merges the processed `spi` values from the CSV exports by `shapeID`.

Main outputs:
- Global Moran's I for SPI, Accessibility, and the scenic-access gap
- LISA cluster maps
- priority tehsil identification for high-scenic / low-access areas
- OLS plus spatial-model diagnostics
- spatial lag/error model selection
- sensitivity analysis for alternative SPI weights
"""
            ),
            code_cell(
                """
from pathlib import Path
import pandas as pd
import geopandas as gpd

from scripts.spatial_analysis_utils import get_analysis_paths, run_full_analysis

PROJECT_ROOT = Path('.').resolve()
PATHS = get_analysis_paths(PROJECT_ROOT)
PATHS
"""
            ),
            md_cell(
                """
## Run Full Analysis

This cell computes the remaining analysis and saves tables, GeoJSON, and plots into:

- `outputs/spatial_disparity_analysis/`
"""
            ),
            code_cell(
                """
results = run_full_analysis(PROJECT_ROOT, save_results=True)

gdf = results['gdf']
global_moran = results['global_moran']
priority_tehsils = results['priority_tehsils']
regression = results['regression']
sensitivity = results['sensitivity']
saved_paths = results['saved_paths']

print('Rows analysed:', len(gdf))
print('Priority tehsils:', len(priority_tehsils))
print('Selected spatial model:', regression['selected_model'])
saved_paths
"""
            ),
            md_cell("## Global Moran's I"),
            code_cell(
                """
global_moran
"""
            ),
            md_cell("## Priority High-Scenic / Low-Access Tehsils"),
            code_cell(
                """
priority_tehsils[
    [
        'shapeName',
        'spi',
        'ai',
        'spi_z',
        'ai_z',
        'gap_index',
        'gap_cluster',
    ]
].head(20)
"""
            ),
            md_cell("## LISA Cluster Counts"),
            code_cell(
                """
pd.DataFrame({
    'SPI LISA': gdf['spi_cluster'].value_counts(),
    'AI LISA': gdf['ai_cluster'].value_counts(),
    'Gap LISA': gdf['gap_cluster'].value_counts(),
}).fillna(0).astype(int)
"""
            ),
            md_cell("## OLS and Spatial Diagnostics"),
            code_cell(
                """
regression['ols_coefficients']
"""
            ),
            code_cell(
                """
regression['lm_tests']
"""
            ),
            code_cell(
                """
pd.DataFrame([regression['residual_moran']])
"""
            ),
            code_cell(
                """
regression['spatial_coefficients']
"""
            ),
            md_cell("## SPI Source Check"),
            code_cell(
                """
gdf[['shapeName', 'spi', 'spi_source']].head(10)
"""
            ),
            md_cell("## Sensitivity Analysis"),
            code_cell(
                """
sensitivity
"""
            ),
            md_cell("## Generated Plots"),
            code_cell(
                """
for key, path in saved_paths.items():
    print(f'{key}: {path}')
"""
            ),
            code_cell(
                """
from IPython.display import Image, display

for key in ['choropleths', 'lisa_maps', 'regression_plot', 'sensitivity_plot']:
    print(key)
    display(Image(filename=str(saved_paths[key])))
"""
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "sds", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
