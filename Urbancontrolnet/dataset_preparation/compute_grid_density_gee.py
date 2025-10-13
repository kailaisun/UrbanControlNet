#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
City-level processing pipeline with two components:

1) DEM → JPEG preparation (normalize, clip, resize, export):
   - Reads city DEM GeoTIFF tiles.
   - Normalizes by subtracting per-tile min, clips at 100 m, scales to [0, 255],
     resizes to 512×512, and writes JPEGs.

2) Grid-based metrics via Google Earth Engine (GHSL):
   - Converts local grid polygons to an Earth Engine FeatureCollection.
   - Computes per-grid metrics using GHSL rasters (built volume/surface/height, population)
     at 100 m resolution.
   - Writes per-city per-grid metrics to GeoJSON.

"""

import os
import re
import time
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import cv2
import rasterio
import ee
from shapely.geometry import box, mapping


# ---------------------------------------------------------------------
# Configuration (adjust paths as needed)
# ---------------------------------------------------------------------
DATA_DIR = "./urban_data/meta_data/"
GHSL_UC_FILE = os.path.join(DATA_DIR, "GHSL_UC_500samples.geojson")

# Input/Output layout (relative, public-safe)
CITY_META_DIR_FMT = os.path.join(DATA_DIR, "meta_data_500samples", "{city}")
CITY_PROC_DIR_FMT = os.path.join(DATA_DIR, "processed_data_500samples", "{city}")
DEM_IN_DIR_NAME = "dem_image"         # inside CITY_PROC_DIR
DEM_OUT_DIR_NAME = "dem_image_jpg"    # inside CITY_PROC_DIR

# Grid input file name pattern (e.g., grid_r0_d0_waterarea.geojson)
GRID_FILE_REGEX = re.compile(r"^grid_r\d+_d\d+_waterarea\.geojson$")

# Metrics output suffix
GRID_METRICS_SUFFIX = "_density_gee.geojson"


# ---------------------------------------------------------------------
# Earth Engine initialization
# ---------------------------------------------------------------------
def ee_initialize() -> None:
    """
    Initialize Earth Engine client.

    Behavior:
        - Tries ee.Initialize(). If credentials are missing, instructs the user to authenticate.
    """
    try:
        ee.Initialize()
    except Exception as e:
        # First-time users must authenticate in a browser, then re-run.
        print("Earth Engine is not initialized. Attempting authentication...")
        try:
            ee.Authenticate()  # Opens browser for OAuth
            ee.Initialize()
        except Exception as e2:
            raise RuntimeError(
                "Failed to initialize Earth Engine. "
                "Please ensure you have an EE account, run ee.Authenticate(), and try again."
            ) from e2


# ---------------------------------------------------------------------
# Part 1: DEM → JPEG conversion utilities
# ---------------------------------------------------------------------
def process_city_dem_to_jpeg(city: str) -> None:
    """
    Convert city DEM tiles (.tif) to normalized 8-bit grayscale JPEGs (512×512).

    Steps:
        - Read first band from each .tif under <CITY_PROC_DIR>/dem_image.
        - Normalize per tile: subtract min, clip values > 100 m, scale to [0,255].
        - Cast to uint8, resize to 512×512, and save as JPEG under dem_image_jpg.

    Parameters
    ----------
    city : str
        City name (directory component).
    """
    city_proc_dir = CITY_PROC_DIR_FMT.format(city=city)
    dem_in_dir = os.path.join(city_proc_dir, DEM_IN_DIR_NAME)
    dem_out_dir = os.path.join(city_proc_dir, DEM_OUT_DIR_NAME)
    os.makedirs(dem_out_dir, exist_ok=True)

    if not os.path.isdir(dem_in_dir):
        # Nothing to process for this city
        return

    for fname in os.listdir(dem_in_dir):
        if not fname.lower().endswith(".tif"):
            continue

        in_path = os.path.join(dem_in_dir, fname)
        out_name = os.path.splitext(fname)[0] + ".jpg"
        out_path = os.path.join(dem_out_dir, out_name)

        try:
            with rasterio.open(in_path) as src:
                dem = src.read(1)  # first band
        except Exception as e:
            print(f"[DEM] Skip unreadable file: {in_path} ({e})")
            continue

        # Per-tile normalization and clipping
        dem_min = float(np.nanmin(dem))
        dem = dem - dem_min
        dem = np.clip(dem, 0, 100)           # cap at 100 m
        dem = (dem / 100.0) * 255.0          # scale to [0, 255]
        dem = dem.astype(np.uint8)

        # Resize and export (grayscale)
        dem_resized = cv2.resize(dem, (512, 512), interpolation=cv2.INTER_AREA)
        ok = cv2.imwrite(out_path, dem_resized)
        if not ok:
            print(f"[DEM] Failed to write JPEG: {out_path}")


# ---------------------------------------------------------------------
# Part 2: Grid → Earth Engine FeatureCollection
# ---------------------------------------------------------------------
def gdf_to_ee_featurecollection(gdf: gpd.GeoDataFrame) -> "ee.FeatureCollection":
    """
    Convert a GeoDataFrame to an Earth Engine FeatureCollection, preserving properties.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input polygons in a geographic CRS (EPSG:4326 recommended).

    Returns
    -------
    ee.FeatureCollection
        FeatureCollection with geometry and original properties per feature.
    """
    features = []
    for _, row in gdf.iterrows():
        geom = mapping(row["geometry"])
        ee_geom = ee.Geometry(geom)
        props = row.drop(labels=["geometry"]).to_dict()
        features.append(ee.Feature(ee_geom, props))
    return ee.FeatureCollection(features)


# ---------------------------------------------------------------------
# Part 3: Batch compute GHSL metrics on grids (via GEE)
# ---------------------------------------------------------------------
def batch_compute_metrics(grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute GHSL-based metrics for all polygons in a grid GeoDataFrame.

    Metrics are computed using Earth Engine reduceRegion calls on the following datasets:
        - Built Volume (JRC/GHSL/P2023A/GHS_BUILT_V/2020)
        - Built Surface (JRC/GHSL/P2023A/GHS_BUILT_S/2020)
        - Built Height (JRC/GHSL/P2023A/GHS_BUILT_H/2018) [mean, stdDev]
        - Population   (JRC/GHSL/P2023A/GHS_POP/2020)

    Aggregation scale is 100 meters.

    Parameters
    ----------
    grid_gdf : GeoDataFrame
        Grid polygons (recommended CRS: EPSG:4326). All properties are retained.

    Returns
    -------
    GeoDataFrame
        A new GeoDataFrame with the same polygons and additional metric fields:
        - built_volume_total, built_volume_nres
        - built_surface_total, built_surface_nres
        - built_height_mean, built_height_std
        - population_count
    """
    # Convert local grid to EE features
    ee_grid_fc = gdf_to_ee_featurecollection(grid_gdf)

    # Load GHSL images once
    built_volume_img = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_V/2020")
    built_surface_img = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020")
    built_height_img = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018")
    population_img = ee.Image("JRC/GHSL/P2023A/GHS_POP/2020")

    def _compute_metrics(feature):
        """
        Earth Engine per-feature reducer. Returns feature with appended metrics.
        """
        # Built volume (sum)
        built_vol = built_volume_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=feature.geometry(),
            scale=100,
            maxPixels=1e13,
        )

        # Built surface (sum)
        built_surf = built_surface_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=feature.geometry(),
            scale=100,
            maxPixels=1e13,
        )

        # Built height (mean, stdDev)
        built_height = built_height_img.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=feature.geometry(),
            scale=100,
            maxPixels=1e13,
        )

        # Population (sum)
        population = population_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=feature.geometry(),
            scale=100,
            maxPixels=1e13,
        )

        # Preserve original properties and append computed metrics
        return feature.set({
            "built_volume_total": built_vol.get("built_volume_total"),
            "built_volume_nres": built_vol.get("built_volume_nres"),
            "built_surface_total": built_surf.get("built_surface"),
            "built_surface_nres": built_surf.get("built_surface_nres"),
            "built_height_mean": built_height.get("built_height_mean"),
            "built_height_std": built_height.get("built_height_stdDev"),
            "population_count": population.get("population_count"),
        })

    # Apply on the server side
    ee_results_fc = ee_grid_fc.map(_compute_metrics)

    # Retrieve results to client (GeoJSON-like dict)
    results = ee_results_fc.getInfo()

    # Convert to GeoDataFrame
    features = results.get("features", [])
    # Pass raw EE features to from_features for accurate geometry decoding
    result_gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    return result_gdf


# ---------------------------------------------------------------------
# Part 4: Orchestration per city
# ---------------------------------------------------------------------
def process_city(city: str) -> None:
    """
    For a given city:
        - Prepare DEM JPEGs.
        - Find grid files and compute GHSL-based metrics for each grid file.
        - Write per-grid metrics to GeoJSON (skips existing outputs).

    Parameters
    ----------
    city : str
        City name (directory component).
    """
    start = time.time()

    # 1) DEM → JPEG conversion (no-op if DEM directory is missing)
    process_city_dem_to_jpeg(city)

    # 2) Grid → metrics via GEE
    city_meta_dir = CITY_META_DIR_FMT.format(city=city)
    city_proc_dir = CITY_PROC_DIR_FMT.format(city=city)
    os.makedirs(city_proc_dir, exist_ok=True)

    if not os.path.isdir(city_meta_dir):
        print(f"[Grid] No meta_data directory for city: {city}")
        return

    grid_files = [
        f for f in os.listdir(city_meta_dir)
        if GRID_FILE_REGEX.match(f)
    ]

    for grid_file in grid_files:
        grid_in_path = os.path.join(city_meta_dir, grid_file)
        out_name = os.path.splitext(grid_file)[0] + GRID_METRICS_SUFFIX
        out_path = os.path.join(city_proc_dir, out_name)

        if os.path.exists(out_path):
            # Skip already processed
            continue

        try:
            grid_gdf = gpd.read_file(grid_in_path)
            # Ensure WGS84 for EE
            grid_gdf = grid_gdf.to_crs(4326)
        except Exception as e:
            print(f"[Grid] Failed to read grid file: {grid_in_path} ({e})")
            continue

        # Compute metrics on GEE
        try:
            metrics_gdf = batch_compute_metrics(grid_gdf)
        except Exception as e:
            print(f"[GEE] Metrics computation failed for {city} :: {grid_file} ({e})")
            continue

        # Basic sanity checks (thresholds per original logic)
        try:
            if "built_surface_total" in metrics_gdf.columns and pd.notnull(metrics_gdf["built_surface_total"]).any():
                if float(metrics_gdf["built_surface_total"].max()) >= 160000:
                    print(f"[Check] Large built_surface_total in {city} :: {grid_file}")
            if metrics_gdf.isna().any().any():
                print(f"[Check] NaN values present in {city} :: {grid_file}")
        except Exception:
            # Do not block export on checks
            pass

        # Export to GeoJSON
        try:
            metrics_gdf.to_file(out_path, driver="GeoJSON")
        except Exception as e:
            print(f"[IO] Failed to write output: {out_path} ({e})")

    print(f"[Done] {city} in {time.time() - start:.2f}s")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    """
    Driver function:
        - Initialize Earth Engine.
        - Load GHSL Urban Center list.
        - Iterate cities and process each.
    """
    ee_initialize()

    try:
        ghsl_uc_gdf = gpd.read_file(GHSL_UC_FILE)
    except Exception as e:
        raise FileNotFoundError(f"Cannot read GHSL UC file: {GHSL_UC_FILE}") from e

    # Loop over all cities listed in the GHSL UC file
    for city_id, city_name, country in zip(
        ghsl_uc_gdf["ID"].tolist(),
        ghsl_uc_gdf["City"].tolist(),
        ghsl_uc_gdf["Country"].tolist(),
    ):
        # The processing functions use only city_name for paths
        process_city(city_name)


if __name__ == "__main__":
    main()
