#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch OSM land-use polygons and building footprints for city boundaries
derived from GHSL Urban Center samples.

This script:
    1. Reads GHSL Urban Center polygons.
    2. Computes appropriate UTM projection per city.
    3. Crops overly large polygons to a fixed-size bounding box.
    4. Buffers boundaries and converts them to WGS84 for OSM queries.
    5. Fetches OSM land-use polygons and building footprints via OSMnx.
    6. Saves results as GeoJSON files.

"""

import os
import time
import math
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import box, Polygon, MultiPolygon

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATA_DIR = "./urban_data/meta_data/"
GHSL_LANDUSE_FILE = os.path.join(DATA_DIR, "GHSL_UC_500samples.geojson")
GHSL_BUILDING_FILE = os.path.join(DATA_DIR, "GHSL_UC_112samples.geojson")
OUT_LANDUSE_DIR = os.path.join(DATA_DIR, "osm_landuse_data")
OUT_BUILDING_DIR = os.path.join(DATA_DIR, "osm_building_data")

os.makedirs(OUT_LANDUSE_DIR, exist_ok=True)
os.makedirs(OUT_BUILDING_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# CRS and Geometry Utilities
# ---------------------------------------------------------------------
def get_utm_crs(boundary_gdf: gpd.GeoDataFrame) -> str:
    """
    Determine the optimal UTM projected CRS (EPSG code) for a given boundary.

    Parameters
    ----------
    boundary_gdf : GeoDataFrame
        City boundary geometry.

    Returns
    -------
    str
        EPSG code string for the best-fitting UTM zone.
    """
    centroid = boundary_gdf.to_crs("EPSG:4326").geometry.unary_union.centroid
    lat, lon = centroid.y, centroid.x
    utm_zone = math.floor((lon + 180) / 6) + 1
    epsg_code = (32600 if lat >= 0 else 32700) + utm_zone
    return f"EPSG:{epsg_code}"


def crop_boundary(boundary_gdf: gpd.GeoDataFrame, city_crs: str) -> gpd.GeoDataFrame:
    """
    Crop a large boundary to a fixed bounding box centered on its centroid.

    Parameters
    ----------
    boundary_gdf : GeoDataFrame
        Boundary geometry.
    city_crs : str
        Projected CRS for the target city.

    Returns
    -------
    GeoDataFrame
        Cropped boundary in the projected CRS.
    """
    if boundary_gdf["Area_Km2"].iloc[0] < 500:
        return boundary_gdf.to_crs(city_crs)

    boundary_gdf = boundary_gdf.to_crs(city_crs)
    centroid = boundary_gdf.geometry.unary_union.centroid
    cx, cy = centroid.x, centroid.y

    minx, maxx = cx - 12500, cx + 12500
    miny, maxy = cy - 12500, cy + 12500

    bbox = box(minx, miny, maxx, maxy)
    bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs=boundary_gdf.crs)
    cropped = gpd.overlay(boundary_gdf, bbox_gdf, how="intersection")
    return cropped


def buffer_city_boundary(boundary_gdf: gpd.GeoDataFrame, city_crs: str):
    """
    Buffer the city boundary and return the bounding polygon in WGS84.

    Parameters
    ----------
    boundary_gdf : GeoDataFrame
        City boundary geometry.
    city_crs : str
        Projected CRS for buffering operations.

    Returns
    -------
    shapely.geometry.Polygon
        Buffered bounding box geometry in WGS84 coordinates.
    """
    boundary_gdf = boundary_gdf.to_crs(city_crs)
    buffered = boundary_gdf.buffer(1000)  # 1 km buffer
    minx, miny, maxx, maxy = buffered.total_bounds
    bbox_polygon = box(minx, miny, maxx, maxy)
    return gpd.GeoSeries([bbox_polygon], crs=city_crs).to_crs(epsg=4326).unary_union


# ---------------------------------------------------------------------
# OSM Data Fetchers
# ---------------------------------------------------------------------
def get_landuse_gdf(boundary_polygon, city_crs: str) -> gpd.GeoDataFrame | None:
    """
    Retrieve OSM land-use polygons within a city boundary.

    Parameters
    ----------
    boundary_polygon : shapely Polygon or MultiPolygon
        Boundary in EPSG:4326 coordinates.
    city_crs : str
        Projected CRS for output geometries.

    Returns
    -------
    GeoDataFrame or None
        Land-use polygons with assigned fclass categories.
    """
    osm_tag_to_fclass = {
        ("landuse", "forest"): "forest",
        ("natural", "wood"): "forest",
        ("leisure", "park"): "park",
        ("leisure", "common"): "park",
        ("landuse", "residential"): "residential",
        ("landuse", "industrial"): "industrial",
        ("landuse", "cemetery"): "cemetery",
        ("landuse", "allotments"): "allotments",
        ("landuse", "meadow"): "meadow",
        ("landuse", "commercial"): "commercial",
        ("leisure", "nature_reserve"): "nature_reserve",
        ("leisure", "recreation_ground"): "recreation_ground",
        ("landuse", "retail"): "retail",
        ("landuse", "military"): "military",
        ("landuse", "quarry"): "quarry",
        ("landuse", "orchard"): "orchard",
        ("landuse", "vineyard"): "vineyard",
        ("landuse", "scrub"): "scrub",
        ("landuse", "grass"): "grass",
        ("natural", "heath"): "heath",
        ("boundary", "national_park"): "national_park",
        ("landuse", "basin"): "basin",
        ("landuse", "village_green"): "village_green",
        ("landuse", "plant_nursery"): "plant_nursery",
        ("landuse", "brownfield"): "brownfield",
        ("landuse", "greenfield"): "greenfield",
        ("landuse", "construction"): "construction",
        ("landuse", "railway"): "railway",
        ("landuse", "farmland"): "farmland",
        ("landuse", "farmyard"): "farmyard",
    }

    tags = {}
    for key, value in osm_tag_to_fclass:
        tags.setdefault(key, []).append(value)

    try:
        gdf = ox.features_from_polygon(boundary_polygon, tags)
        if gdf is None or gdf.empty:
            print("No land-use features found.")
            return None

        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

        def map_fclass(row):
            for (k, v), fclass in osm_tag_to_fclass.items():
                if row.get(k) == v:
                    return fclass
            return None

        gdf["fclass"] = gdf.apply(map_fclass, axis=1)
        gdf = gdf[gdf["fclass"].notnull()]
        gdf = gdf.to_crs(city_crs)

        print(f"Retrieved {len(gdf)} land-use polygons.")
        return gdf

    except ox._errors.InsufficientResponseError as e:
        print(f"OSMnx error: {e}")
    except Exception as e:
        print(f"Unexpected error while fetching land-use data: {e}")

    return None


def get_building_gdf(boundary_polygon, city_crs: str) -> gpd.GeoDataFrame | None:
    """
    Retrieve OSM building footprints within a city boundary.

    Parameters
    ----------
    boundary_polygon : shapely Polygon or MultiPolygon
        Boundary in EPSG:4326 coordinates.
    city_crs : str
        Projected CRS for output geometries.

    Returns
    -------
    GeoDataFrame or None
        Building polygons with a non-empty 'function' field.
    """
    tags = {"building": True}

    try:
        gdf = ox.features_from_polygon(boundary_polygon, tags)
        if gdf is None or gdf.empty:
            print("No building features found.")
            return None

        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
        gdf = gdf[gdf["building"].notnull() & (gdf["building"] != "yes") & (gdf["building"] != "roof")]
        gdf = gdf.rename(columns={"building": "function"})

        keep_cols = ["function", "geometry"]
        gdf = gdf[[c for c in keep_cols if c in gdf.columns]]
        gdf = gdf.loc[:, ~gdf.columns.duplicated()]
        gdf = gdf.to_crs(city_crs)

        print(f"Retrieved {len(gdf)} building polygons.")
        return gdf

    except ox._errors.InsufficientResponseError as e:
        print(f"OSMnx error: {e}")
    except Exception as e:
        print(f"Unexpected error while fetching building data: {e}")

    return None


# ---------------------------------------------------------------------
# Processing Pipelines
# ---------------------------------------------------------------------
def run_landuse_pipeline():
    """Download and export OSM land-use data for each GHSL city."""
    print(f"Reading GHSL file: {GHSL_LANDUSE_FILE}")
    all_gdf = gpd.read_file(GHSL_LANDUSE_FILE)
    print(f"CRS: {all_gdf.crs}")

    for cid, city, country, area in zip(
        all_gdf["ID"], all_gdf["City"], all_gdf["Country"], all_gdf["Area_Km2"]
    ):
        out_path = os.path.join(OUT_LANDUSE_DIR, f"landuse_{cid}.geojson")
        if os.path.exists(out_path):
            continue

        print(f"\n[Land-use] {cid} | {city} | {country}")
        boundary = all_gdf.loc[all_gdf["ID"] == cid].copy()
        city_crs = get_utm_crs(boundary)
        boundary = crop_boundary(boundary, city_crs)
        boundary_polygon = buffer_city_boundary(boundary, city_crs)

        start = time.time()
        gdf = get_landuse_gdf(boundary_polygon, city_crs)
        if gdf is not None:
            gdf.to_file(out_path, driver="GeoJSON")
        print(f"{city}: land-use extraction completed in {time.time() - start:.2f}s.")


def run_building_pipeline():
    """Download and export OSM building footprint data for each GHSL city."""
    print(f"Reading GHSL file: {GHSL_BUILDING_FILE}")
    all_gdf = gpd.read_file(GHSL_BUILDING_FILE)
    print(f"CRS: {all_gdf.crs}")

    for cid, city, country, area in zip(
        all_gdf["ID"], all_gdf["City"], all_gdf["Country"], all_gdf["Area_Km2"]
    ):
        out_path = os.path.join(OUT_BUILDING_DIR, f"building_{cid}.geojson")
        if os.path.exists(out_path):
            continue

        print(f"\n[Building] {cid} | {city} | {country}")
        boundary = all_gdf.loc[all_gdf["ID"] == cid].copy()
        city_crs = get_utm_crs(boundary)
        boundary = crop_boundary(boundary, city_crs)
        boundary_polygon = buffer_city_boundary(boundary, city_crs)

        start = time.time()
        gdf = get_building_gdf(boundary_polygon, city_crs)
        if gdf is not None:
            gdf.to_file(out_path, driver="GeoJSON")
        print(f"{city}: building extraction completed in {time.time() - start:.2f}s.")


# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------
def main():
    run_landuse_pipeline()
    run_building_pipeline()


if __name__ == "__main__":
    main()
