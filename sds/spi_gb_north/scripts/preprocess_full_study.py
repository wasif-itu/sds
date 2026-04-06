#!/usr/bin/env python3
"""Preprocess full-study datasets for SPI/AI analysis.

This script prepares analysis-ready tehsil-level inputs using:
- ADM3 boundaries (GeoBoundaries)
- DEM raster
- 2024 land cover raster (best-overlap tile is auto-selected)
- OSM roads (primary/secondary)

Outputs are written to:
- data/interim/   (cleaned vectors + derived rasters)
- data/processed/ (tehsil summary table + geospatial layer)

Snow/ice is intentionally excluded in this phase and can be added later.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy import ndimage
from shapely.geometry import box

TARGET_CRS = "EPSG:32643"
FOREST_CODE = 10
WATER_CODE = 80


@dataclass
class TemplateGrid:
    transform: rasterio.Affine
    width: int
    height: int
    crs: str
    resolution_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw data for full-study SPI/AI analysis"
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Path to spi_gb_north root (default: parent of this script)",
    )
    parser.add_argument(
        "--resolution-m",
        type=float,
        default=100.0,
        help="Target raster resolution in meters (default: 100)",
    )
    return parser.parse_args()


def fix_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def choose_best_landcover_tile(
    landcover_files: Iterable[Path],
    boundary_wgs84: gpd.GeoDataFrame,
) -> Path:
    boundary_union = boundary_wgs84.geometry.unary_union
    best_file = None
    best_overlap = -1.0

    for path in landcover_files:
        with rasterio.open(path) as src:
            left, bottom, right, top = src.bounds
            footprint = gpd.GeoDataFrame(
                geometry=[box(left, bottom, right, top)],
                crs=src.crs,
            ).to_crs("EPSG:4326")
            overlap = footprint.geometry.iloc[0].intersection(boundary_union).area
            if overlap > best_overlap:
                best_overlap = overlap
                best_file = path

    if best_file is None:
        raise FileNotFoundError("No landcover raster found in raw/landcover")

    return best_file


def create_template_grid(
    boundary_utm: gpd.GeoDataFrame,
    resolution_m: float,
) -> TemplateGrid:
    minx, miny, maxx, maxy = boundary_utm.total_bounds

    width = int(np.ceil((maxx - minx) / resolution_m))
    height = int(np.ceil((maxy - miny) / resolution_m))

    # Snap top-left to exact boundary min/max at target resolution.
    transform = from_origin(minx, maxy, resolution_m, resolution_m)

    return TemplateGrid(
        transform=transform,
        width=width,
        height=height,
        crs=TARGET_CRS,
        resolution_m=resolution_m,
    )


def reproject_raster_to_template(
    src_path: Path,
    grid: TemplateGrid,
    resampling: Resampling,
    dst_nodata: float,
) -> np.ndarray:
    with rasterio.open(src_path) as src:
        dst = np.full((grid.height, grid.width), dst_nodata, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )
    return dst


def write_geotiff(
    out_path: Path,
    array: np.ndarray,
    grid: TemplateGrid,
    nodata: float,
    dtype: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        width=grid.width,
        height=grid.height,
        count=1,
        dtype=dtype,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(array.astype(dtype), 1)


def compute_tri(dem: np.ndarray) -> np.ndarray:
    # Riley-style TRI proxy: root mean square elevation difference to 8 neighbors.
    dem = dem.astype(np.float32)
    valid_center = np.isfinite(dem)

    acc = np.zeros_like(dem, dtype=np.float32)
    count = np.zeros_like(dem, dtype=np.float32)

    directions: Tuple[Tuple[int, int], ...] = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    rows, cols = dem.shape

    for dy, dx in directions:
        shifted = np.roll(np.roll(dem, dy, axis=0), dx, axis=1)
        valid = valid_center & np.isfinite(shifted)

        if dy == -1:
            valid[-1, :] = False
        if dy == 1:
            valid[0, :] = False
        if dx == -1:
            valid[:, -1] = False
        if dx == 1:
            valid[:, 0] = False

        diff2 = np.zeros_like(dem, dtype=np.float32)
        diff2[valid] = (shifted[valid] - dem[valid]) ** 2
        acc += diff2
        count += valid.astype(np.float32)

    tri = np.full_like(dem, np.nan, dtype=np.float32)
    ok = count > 0
    tri[ok] = np.sqrt(acc[ok] / count[ok])
    return tri


def linestring_primary_secondary_mask(value: object) -> bool:
    if value is None:
        return False
    text = str(value).lower()
    return ("primary" in text) or ("secondary" in text)


def build_tehsil_label_raster(
    tehsils: gpd.GeoDataFrame,
    grid: TemplateGrid,
) -> np.ndarray:
    shapes = ((geom, idx + 1) for idx, geom in enumerate(tehsils.geometry))
    labels = features.rasterize(
        shapes=shapes,
        out_shape=(grid.height, grid.width),
        transform=grid.transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )
    return labels


def zonal_mean_from_labels(
    values: np.ndarray,
    labels: np.ndarray,
    n_labels: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = (labels > 0) & np.isfinite(values)
    if valid_mask is not None:
        mask &= valid_mask

    sums = np.bincount(
        labels[mask],
        weights=values[mask],
        minlength=n_labels + 1,
    )
    counts = np.bincount(labels[mask], minlength=n_labels + 1)

    out = np.full(n_labels, np.nan, dtype=np.float64)
    nz = counts[1:] > 0
    out[nz] = sums[1:][nz] / counts[1:][nz]
    return out


def zonal_percent_class_from_labels(
    class_mask: np.ndarray,
    labels: np.ndarray,
    n_labels: int,
    denominator_mask: np.ndarray,
) -> np.ndarray:
    denom_counts = np.bincount(
        labels[(labels > 0) & denominator_mask],
        minlength=n_labels + 1,
    )
    class_counts = np.bincount(
        labels[(labels > 0) & denominator_mask & class_mask],
        minlength=n_labels + 1,
    )

    pct = np.full(n_labels, np.nan, dtype=np.float64)
    nz = denom_counts[1:] > 0
    pct[nz] = (class_counts[1:][nz] / denom_counts[1:][nz]) * 100.0
    return pct


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent

    raw_dir = project_root / "data" / "raw"
    interim_dir = project_root / "data" / "interim"
    processed_dir = project_root / "data" / "processed"
    metadata_dir = project_root / "data" / "metadata"

    for d in [interim_dir, processed_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    admin_path = raw_dir / "admin_boundaries" / "geoBoundaries-PAK-ADM3.geojson"
    dem_path = raw_dir / "dem" / "dem.tif"
    roads_path = raw_dir / "roads" / "spi_gb_north.gpkg"
    clipping_boundary_path = raw_dir / "roads" / "clipping_boundary.geojson"

    landcover_files = sorted((raw_dir / "landcover").glob("*.tif"))
    if not landcover_files:
        raise FileNotFoundError("No .tif files found in data/raw/landcover")

    print("[1/7] Loading study boundary and tehsils...")
    boundary = gpd.read_file(clipping_boundary_path)
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    boundary_wgs84 = boundary.to_crs("EPSG:4326")

    tehsils = gpd.read_file(admin_path)
    if tehsils.crs is None:
        tehsils = tehsils.set_crs("EPSG:4326")

    tehsils = fix_geometries(tehsils)
    tehsils = tehsils[tehsils.intersects(boundary_wgs84.geometry.unary_union)].copy()
    tehsils = tehsils.to_crs(TARGET_CRS)

    tehsil_out = interim_dir / "tehsils_north_32643.gpkg"
    tehsils.to_file(tehsil_out, layer="tehsils_north", driver="GPKG")

    print(f"Selected tehsils: {len(tehsils)}")

    print("[2/7] Building target raster grid...")
    boundary_utm = boundary_wgs84.to_crs(TARGET_CRS)
    grid = create_template_grid(boundary_utm, args.resolution_m)

    region_mask = features.geometry_mask(
        [boundary_utm.geometry.unary_union],
        out_shape=(grid.height, grid.width),
        transform=grid.transform,
        invert=True,
    )

    print("[3/7] Selecting and preprocessing 2024 land cover...")
    best_landcover = choose_best_landcover_tile(landcover_files, boundary_wgs84)
    print(f"Using landcover tile: {best_landcover.name}")

    landcover = reproject_raster_to_template(
        src_path=best_landcover,
        grid=grid,
        resampling=Resampling.nearest,
        dst_nodata=0,
    )
    landcover[~region_mask] = 0

    lc_out = interim_dir / "worldcover_2024_32643_resampled_100m.tif"
    write_geotiff(lc_out, landcover, grid, nodata=0, dtype="uint8")

    forest_mask = landcover == FOREST_CODE
    water_mask = landcover == WATER_CODE
    write_geotiff(interim_dir / "forest_mask_2024_32643_100m.tif", forest_mask.astype(np.uint8), grid, nodata=0, dtype="uint8")
    write_geotiff(interim_dir / "water_mask_2024_32643_100m.tif", water_mask.astype(np.uint8), grid, nodata=0, dtype="uint8")

    print("[4/7] Preprocessing DEM and deriving TRI...")
    dem = reproject_raster_to_template(
        src_path=dem_path,
        grid=grid,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
    )
    dem[~region_mask] = np.nan

    tri = compute_tri(dem)
    tri[~region_mask] = np.nan

    write_geotiff(interim_dir / "dem_32643_100m_clip.tif", dem, grid, nodata=np.nan, dtype="float32")
    write_geotiff(interim_dir / "tri_derived_32643_100m.tif", tri, grid, nodata=np.nan, dtype="float32")

    print("[5/7] Preprocessing roads and distance-to-roads raster...")
    roads = gpd.read_file(roads_path)
    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")

    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    if "highway" in roads.columns:
        roads = roads[roads["highway"].apply(linestring_primary_secondary_mask)].copy()

    roads = roads.to_crs(TARGET_CRS)
    roads = gpd.clip(roads, boundary_utm)

    roads_out = interim_dir / "roads_primary_secondary_32643.gpkg"
    roads.to_file(roads_out, layer="roads_primary_secondary", driver="GPKG")

    road_raster = features.rasterize(
        ((geom, 1) for geom in roads.geometry),
        out_shape=(grid.height, grid.width),
        transform=grid.transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    road_raster[~region_mask] = 0

    # Euclidean distance transform in meters (0 for road pixels).
    dist_to_road = ndimage.distance_transform_edt(road_raster == 0, sampling=grid.resolution_m)
    dist_to_road = dist_to_road.astype(np.float32)
    dist_to_road[~region_mask] = np.nan

    write_geotiff(interim_dir / "distance_to_roads_32643_100m.tif", dist_to_road, grid, nodata=np.nan, dtype="float32")

    print("[6/7] Computing tehsil-level zonal metrics...")
    labels = build_tehsil_label_raster(tehsils, grid)
    n_tehsils = len(tehsils)

    lc_valid = (landcover > 0) & region_mask

    tri_mean = zonal_mean_from_labels(tri, labels, n_tehsils)
    dist_mean = zonal_mean_from_labels(dist_to_road, labels, n_tehsils)
    forest_pct = zonal_percent_class_from_labels(forest_mask, labels, n_tehsils, lc_valid)
    water_pct = zonal_percent_class_from_labels(water_mask, labels, n_tehsils, lc_valid)

    # Road density (km of primary/secondary roads per km^2 tehsil area).
    tehsils_metrics = tehsils[["shapeName", "shapeID", "geometry"]].copy()
    tehsils_metrics["tehsil_area_km2"] = tehsils_metrics.geometry.area / 1_000_000.0

    if roads.empty:
        tehsils_metrics["road_length_km"] = 0.0
    else:
        intersection = gpd.overlay(
            roads[["geometry"]],
            tehsils_metrics[["shapeID", "geometry"]],
            how="intersection",
        )
        if intersection.empty:
            road_len = pd.Series(dtype=float)
        else:
            intersection["seg_km"] = intersection.geometry.length / 1000.0
            road_len = intersection.groupby("shapeID")["seg_km"].sum()

        tehsils_metrics["road_length_km"] = tehsils_metrics["shapeID"].map(road_len).fillna(0.0)

    tehsils_metrics["road_density_km_per_km2"] = (
        tehsils_metrics["road_length_km"] / tehsils_metrics["tehsil_area_km2"]
    )

    tehsils_metrics["tri_mean"] = tri_mean
    tehsils_metrics["forest_pct"] = forest_pct
    tehsils_metrics["water_pct"] = water_pct
    tehsils_metrics["dist_to_roads_mean_m"] = dist_mean

    print("[7/7] Building SPI and AI outputs...")
    tri_z = zscore(tehsils_metrics["tri_mean"])
    forest_z = zscore(tehsils_metrics["forest_pct"])
    water_z = zscore(tehsils_metrics["water_pct"])

    # Snow/ice is deferred. We renormalize by the active weight sum (0.85).
    tehsils_metrics["spi_no_snow"] = (
        (0.4 * tri_z) + (0.25 * forest_z) + (0.2 * water_z)
    ) / 0.85

    road_density_z = zscore(tehsils_metrics["road_density_km_per_km2"])
    dist_road_z = zscore(tehsils_metrics["dist_to_roads_mean_m"])
    tehsils_metrics["ai"] = (0.5 * road_density_z) - (0.5 * dist_road_z)

    # Priority flag for phase-1 screening: above-median scenic, below-median access.
    spi_med = tehsils_metrics["spi_no_snow"].median()
    ai_med = tehsils_metrics["ai"].median()
    tehsils_metrics["priority_high_scenic_low_access"] = (
        (tehsils_metrics["spi_no_snow"] > spi_med)
        & (tehsils_metrics["ai"] < ai_med)
    )

    out_geojson = processed_dir / "tehsil_spi_ai_fullstudy.geojson"
    out_csv = processed_dir / "tehsil_spi_ai_fullstudy.csv"

    tehsils_metrics.to_file(out_geojson, driver="GeoJSON")
    tehsils_metrics.drop(columns=["geometry"]).to_csv(out_csv, index=False)

    summary_txt = metadata_dir / "preprocessing_run_summary.txt"
    summary_txt.write_text(
        "\n".join(
            [
                f"tehsils_selected={len(tehsils_metrics)}",
                f"target_crs={TARGET_CRS}",
                f"target_resolution_m={grid.resolution_m}",
                f"landcover_selected={best_landcover.name}",
                "snow_ice_status=deferred",
                f"outputs_geojson={out_geojson}",
                f"outputs_csv={out_csv}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Preprocessing complete.")
    print(f"- {out_geojson}")
    print(f"- {out_csv}")


if __name__ == "__main__":
    main()
