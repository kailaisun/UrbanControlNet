import subprocess
import geopandas as gpd
import pandas as pd
import json

import os
import time
#import cv2
import numpy as np
import re
import math
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
#from shapely.ops import unary_union
from concurrent.futures import ThreadPoolExecutor
from shapely.strtree import STRtree
from shapely import union_all
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import sys
os.makedirs("landuse_road_logs", exist_ok=True)
f = open("landuse_road_logs/all_output.log", "a", encoding="utf-8")
sys.stdout = f
sys.stderr = f


road_fclass = {
    'rd_major': ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link'],
    'rd_minor': ['unclassified', 'residentail', 'living_street', 'service', 'services'], # exclude 'busway', 'pedestrian', 'closed', 'emergency_bay', 'rest_area'
}
# Define priority list for highway types
priority_highways = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'residentail', 'living_street', 'service', 'services', 'unclassified']


landuse_types = ['residential', 'commercial', 'industrial', 'service', 'parking', 'recreational', 'agriculture', 'natural']

landuse_fclass = {
    # Residential 
    'residential'       : 'residential',
    
    # Industrial
    'industrial'        : 'industrial',
    'quarry'            : 'industrial',
    
    # Commercial
    'commercial'        : 'commercial',
    'retail'            : 'commercial',

    # Service
    'military'          : 'service',
    
    # Parking
    'parking'           : 'parking',
    
    # Recreational
    'allotments'        : 'recreational',
    'cemetery'          : 'recreational',
    'park'              : 'recreational',
    'recreation_ground' : 'recreational',
    
    # Agriculture
    'farmland'          : 'agriculture',
    'farmyard'          : 'agriculture',
    'meadow'            : 'agriculture',
    'orchard'           : 'agriculture',
    'plant_nursery'     : 'agriculture',
    'vineyard'          : 'agriculture',
    
    # Natural
    'forest'            : 'natural',
    'grass'             : 'natural',
    'heath'             : 'natural',
    'national_park'     : 'natural',
    'nature_reserve'    : 'natural',
    'scrub'             : 'natural',
    'village_green'     : 'natural',
}

building_to_landuse_map = {
    # Residential
    'house'                  : ['residential'],
    'residential'            : ['residential'],
    'apartments'             : ['residential'],
    'detached'               : ['residential'],
    'terrace'                : ['residential'],
    'semidetached_house'     : ['residential'],
    'allotment_house'        : ['residential'],
    'bungalow'               : ['residential'],
    'hut'                    : ['residential'],
    'duplex'                 : ['residential'],
    'dormitory'              : ['residential'],
    'cabin'                  : ['residential'],
    'houseboat'              : ['residential'],
    'mobile_home'            : ['residential'],
    
    # Industrial
    'shed'                   : ['industrial'],
    'industrial'             : ['industrial'],
    'warehouse'              : ['industrial'],
    'storage_tank'           : ['industrial'],
    'hangar'                 : ['industrial'],
    'silo'                   : ['industrial'],
    'factory'                : ['industrial'],
    'manufacture'            : ['industrial'],
    
    # Commercial
    'commercial'             : ['commercial'],
    'retail'                 : ['commercial'],
    'office'                 : ['commercial'],
    'hotel'                  : ['commercial'],
    'kiosk'                  : ['commercial'],
    'stadium'                : ['commercial'],
    'sports_centre'          : ['commercial'],
    'supermarket'            : ['commercial'],
    'bank'                   : ['commercial'],
    'boathouse'              : ['commercial'],
    'shop'                   : ['commercial'],
    'village_office'         : ['commercial'],
    'marketplace'            : ['commercial'],
    'restaurant'             : ['commercial'],
    'museum'                 : ['commercial'],
    'pub'                    : ['commercial'],
    'community_group_office' : ['commercial'],
    
    # Mixed (Multiple Mappings)
    'commercial;residential': ['commercial', 'residential'],
    
    # Service
    'school'                 : ['service'],
    'university'             : ['service'],
    'service'                : ['service'],
    'public'                 : ['service'],
    'hospital'               : ['service'],
    'church'                 : ['service'],
    'kindergarten'           : ['service'],
    'mosque'                 : ['service'],
    'college'                : ['service'],
    'train_station'          : ['service'],
    'temple'                 : ['service'],
    'civic'                  : ['service'],
    'transportation'         : ['service'],
    'government'             : ['service'],
    'chapel'                 : ['service'],
    'shrine'                 : ['service'],
    'utility'                : ['service'],
    'fire_station'           : ['service'],
    'government_office'      : ['service'],
    'pavilion'               : ['service'],
    'sports_hall'            : ['service'],
    'clinic'                 : ['service'],
    'police'                 : ['service'],
    'cathedral'              : ['service'],
    'religious'              : ['service'],
    
    # Parking
    'garage'                 : ['parking'],
    'carport'                : ['parking'],
    'garages'                : ['parking'],
    'parking'                : ['parking'],
    'parking_entrance'       : ['parking'],
    
    # Agriculture
    'greenhouse'             : ['agriculture'],
    'farm_auxiliary'         : ['agriculture'],
    'barn'                   : ['agriculture'],
    'farm'                   : ['agriculture'],
    'stable'                 : ['agriculture'],
    'cowshed'                : ['agriculture'],
    
}

standard_columns = [
    "row",
    "col",
    "r",
    "d",
    "left_x",
    "top_y",
    "population_count",
    "built_height_mean",
    "built_height_std",
    "built_surface_nres",
    "built_surface_total",
    "built_volume_nres",
    "built_volume_total",
    "water_area_ratio",
    "rd_major_length",
    "rd_minor_length",
    "total_road_length",
    "road_density",
    "residential_area",
    "commercial_area",
    "industrial_area",
    "service_area",
    "parking_area",
    "recreational_area",
    "agriculture_area",
    "natural_area",
    "cellarea",
    "residential_ratio",
    "commercial_ratio",
    "industrial_ratio",
    "service_ratio",
    "parking_ratio",
    "recreational_ratio",
    "agriculture_ratio",
    "natural_ratio",
    "lu_ratio_sum",
    "lu_exceeded"
]


def convert_geojson_to_gpkg(input_path, output_path, fields=None):
    """
    Convert a GeoJSON file to GeoPackage format, selecting only specified fields.
    Only runs conversion if the output file does not already exist.
    """
    if not os.path.exists(output_path):
        print(f"Converting {input_path} to {output_path}...")
        start_time = time.time()

        cmd = [
            "ogr2ogr",
            "-f", "GPKG",
            output_path,
            input_path, 
            "-skipfailures"
        ]
        if fields:
            cmd += ["-select", ",".join(fields)]

        subprocess.run(cmd, check=True)

        elapsed = time.time() - start_time
        print(f"Conversion completed in {elapsed:.2f} seconds.")
    else:
        print(f"Skipping conversion: {output_path} already exists.")


# Compute UTM CRS for Projection
def get_utm_crs(boundary_gdf):
    #centroid = boundary_gdf.to_crs("EPSG:4326").geometry.unary_union.centroid
    centroid = union_all(boundary_gdf.to_crs("EPSG:4326").geometry).centroid
    lat, lon = centroid.y, centroid.x
    utm_zone = math.floor((lon + 180) / 6) + 1
    epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
    return f"EPSG:{epsg_code}"

#   Read a road shapefile/GPKG, count exactly overlapping road segments (same geometry),
#   and return the deduplicated GeoDataFrame.
def read_file_and_drop_duplicate(file_path, projected_crs):
    gdf = gpd.read_file(file_path).to_crs(projected_crs)
    duplicated_mask = gdf.geometry.duplicated(keep='first')
    dup_gdf = gdf[duplicated_mask].copy()

    deduplicated_gdf = gdf.drop_duplicates(subset='geometry', keep='first').copy()
    total_duplicates = len(dup_gdf)

    print(f"📦 Found {total_duplicates} duplicated elements. Original: {len(gdf)}, Deduplicated: {len(deduplicated_gdf)}")

    return total_duplicates, dup_gdf, deduplicated_gdf


def read_road_gdf(road_path, projected_crs):
    #rd_gdf = gpd.read_file(road_path).to_crs(projected_crs)
    _, _, rd_gdf = read_file_and_drop_duplicate(road_path, projected_crs)


    def process_highway(value):
        # If the value is a list, process it
        if value.startswith('['):
            value = eval(value)  # Convert string to list
            # Check if any element in the list is in the priority list
            for highway_type in priority_highways:
                if highway_type in value:
                    return highway_type
            # If no elements match the priority list, return the first one in the list
            return value[0]
        # If the value is not a list, return it as is
        return value
    # print(rd_gdf['highway'].unique())
    # Apply the processing function to the 'highway' column
    rd_gdf['highway'] = rd_gdf['highway'].apply(process_highway)
    # print(rd_gdf['highway'].unique())


    fclass2level = dict()
    for level in road_fclass:
        for fclass in road_fclass[level]:
            fclass2level[fclass] = level
    rd_gdf['highway'] = rd_gdf['highway'].apply(lambda loc: fclass2level.get(loc, None))
    rd_gdf = rd_gdf.dropna(subset = ['highway'])

    # Group by 'fclass'
    rd_grouped = rd_gdf.groupby('highway')

    # Function to safely get a group as a GeoDataFrame
    def get_group_as_gdf(group_name):
        if group_name in rd_grouped.groups:
            return rd_grouped.get_group(group_name)
        else:
            return gpd.GeoDataFrame(columns=rd_gdf.columns, crs=rd_gdf.crs)

    rd_list = dict()
    for rd_type in road_fclass.keys():
        rd_type_specific = get_group_as_gdf(rd_type)

        if rd_type_specific.empty:
            continue  # Skip if no geometries for this road type

        rd_list[rd_type] = rd_type_specific # gpd.GeoDataFrame(geometry=rd_type_specific, crs=rd_gdf.crs)

    return rd_list



def dedupe_landuse_dict(landuse_gdf_dict):
    """
    For each landuse category in the input dict, merge all geometries into
    non-overlapping polygons (using unary_union), and return a new dict
    with cleaned GeoDataFrames.
    """
    deduped = {}
    for category, gdf in landuse_gdf_dict.items():
        if gdf is None or gdf.empty:
            deduped[category] = gdf
            continue
        
        # Merge all geometries into a single MultiPolygon (removes overlaps)
        #merged = gdf.geometry.unary_union
        merged = union_all(gdf.geometry)
        
        # If it's a MultiPolygon, split into list; else wrap single Polygon
        geoms = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        
        # Create cleaned GeoDataFrame
        deduped[category] = gpd.GeoDataFrame(
            {"geometry": geoms},
            crs=gdf.crs
        )
    return deduped



def generate_landuse_complement(
    buildings_gdf: gpd.GeoDataFrame,
    landuse_gdf: gpd.GeoDataFrame,
    building_to_landuse_map: dict,
    building_type_col: str = "function",
    landuse_attr: str = "landuse",
    buffer_distance: float = 50,
    target_crs: str = "EPSG:3857"
) -> gpd.GeoDataFrame:
    """
    From a buildings layer and an existing landuse layer, create a 'landuse_complement'
    GeoDataFrame consisting of those buildings that fall in gaps (no landuse),
    mapped to landuse classes and dissolved by proximity.

    Parameters:
        buildings_gdf: GeoDataFrame of building footprints with a type column.
        landuse_gdf:   GeoDataFrame of existing landuse polygons.
        building_to_landuse_map: dict mapping building types → landuse class.
        building_type_col: name of the column in buildings_gdf with the building type.
        landuse_attr:      name of the resulting landuse attribute for the complement.
        buffer_distance:   distance (in CRS units) for proximity dissolve.
        target_crs:        CRS to reproject before buffering/union (e.g. "EPSG:3857").

    Returns:
        GeoDataFrame with columns [landuse_attr, geometry], where geometries are
        the dissolved complement areas for each landuse class.
    """

    if buildings_gdf.empty:
        raise ValueError("The input buildings_gdf is empty. Cannot generate landuse complement.")

    if building_type_col not in buildings_gdf.columns:
        raise KeyError(f"The expected column '{building_type_col}' is missing from buildings_gdf. "
                       f"Available columns: {list(buildings_gdf.columns)}")
    
    # 1) filter out buildings without a valid type
    b = buildings_gdf.loc[~buildings_gdf[building_type_col].isna()].copy()

    # 2) find buildings NOT covered by any landuse
    joined = gpd.sjoin(
        b, landuse_gdf,
        how="left", predicate="intersects"
    )
    complement = joined[joined["index_right"].isna()].copy()

    # 3) map building type → landuse class(es) - allow multiple mappings (list output)
    #complement[landuse_attr] = complement[building_type_col].map(building_to_landuse_map)
    complement[landuse_attr] = complement[building_type_col].apply(
            lambda x: building_to_landuse_map.get(x, [])
        )
    complement = complement.explode(landuse_attr).reset_index(drop=True) 
    complement = complement[complement[landuse_attr].notna()]


    # 4) project to target CRS for metric operations
    complement = complement.to_crs(target_crs)
    #print(complement[landuse_attr].apply(type).value_counts())

    # 5) dissolve by type & proximity
    dissolved_parts = []
    for cls, group in complement.groupby(landuse_attr):
        # buffer half distance so features within `buffer_distance` merge
        buf = group.geometry.buffer(buffer_distance / 2)

        try:
            merged = union_all(buf)
        except Exception as e:
            print(f"   ❌ Error in union_all() for class {cls}: {e}")
            continue

        if hasattr(merged, "geoms"):
            geoms = list(merged.geoms)
        else:
            geoms = [merged]

        for geom in geoms:
            if isinstance(geom, BaseGeometry) and not geom.is_empty:
                dissolved_parts.append({
                    landuse_attr: cls,
                    "geometry": geom
                })

        for geom in geoms:
            dissolved_parts.append({landuse_attr: cls, "geometry": geom})

        #result = gpd.GeoDataFrame(dissolved_parts, geometry="geometry", crs=target_crs)

    if dissolved_parts:
        result = gpd.GeoDataFrame(dissolved_parts, geometry="geometry", crs=target_crs)
    else:
        print("\n⚠️ No geometries generated after dissolve. Returning empty GeoDataFrame.")
        result = gpd.GeoDataFrame(columns=[landuse_attr, "geometry"], geometry="geometry", crs=target_crs)

    return result


def read_landuse_gdf(landuse_path, projected_crs):
    #landuse = gpd.read_file(landuse_path)
    #if landuse.crs is None:
    #    print("CRS not defined in file, assigning EPSG:4326 (GeoJSON default)")
    #    landuse.set_crs(epsg=4326, inplace=True)
    #landuse = landuse.to_crs(projected_crs)
    _, _, landuse = read_file_and_drop_duplicate(landuse_path, projected_crs)



    landuse['landuse'] = landuse['fclass'].map(landuse_fclass)
    landuse = landuse.dropna(subset=['landuse'])
    landuse_grouped = landuse.groupby('landuse')

    # Function to safely get a group as a GeoDataFrame
    def get_group_as_gdf(group_name):
        if group_name in landuse_grouped.groups:
            return landuse_grouped.get_group(group_name)
        else:
            return gpd.GeoDataFrame(columns=landuse.columns, crs=landuse.crs)

    landuse_list = dict()

    for lu_type in landuse_types:
        lu_gdf = get_group_as_gdf(lu_type)
        if lu_gdf.empty:
            continue
        landuse_list[lu_type] = lu_gdf

    return landuse_list


def read_landuse_building_gdf(landuse_path, building_path, projected_crs):
    #landuse = gpd.read_file(landuse_path)
    #if landuse.crs is None:
    #    print("CRS not defined in file, assigning EPSG:4326 (GeoJSON default)")
    #    landuse.set_crs(epsg=4326, inplace=True)
    #landuse = landuse.to_crs(projected_crs)
    _, _, landuse = read_file_and_drop_duplicate(landuse_path, projected_crs)
    _, _, building = read_file_and_drop_duplicate(building_path, projected_crs)

    try:
        landuse_complement_dissolved = generate_landuse_complement(
            buildings_gdf=building,
            landuse_gdf=landuse,
            building_to_landuse_map=building_to_landuse_map,
            buffer_distance=50,
            target_crs=projected_crs
        )
    except (ValueError, KeyError) as e:
        print(f"❌ Error generating landuse complement.")
        landuse_complement_dissolved = gpd.GeoDataFrame(columns=["landuse", "geometry"], geometry="geometry", crs=projected_crs)


    landuse['landuse'] = landuse['fclass'].map(landuse_fclass)
    landuse = landuse.dropna(subset=['landuse'])

    landuse_complement_dissolved = landuse_complement_dissolved.dropna(
        subset=['landuse'])

    combined = gpd.GeoDataFrame(
        pd.concat([
            landuse[['landuse', 'geometry']],
            landuse_complement_dissolved[['landuse', 'geometry']]
        ], ignore_index=True),
        crs=landuse.crs
    )
    
    combined_dissolved = combined.dissolve(by='landuse')
    landuse_grouped = combined_dissolved.groupby('landuse')

    # Function to safely get a group as a GeoDataFrame
    def get_group_as_gdf(group_name):
        if group_name in landuse_grouped.groups:
            return landuse_grouped.get_group(group_name)
        else:
            return gpd.GeoDataFrame(columns=landuse.columns, crs=landuse.crs)

    landuse_list = dict()

    for lu_type in landuse_types:
        lu_gdf = get_group_as_gdf(lu_type)
        if lu_gdf.empty:
            continue
        landuse_list[lu_type] = lu_gdf

    return landuse_list



def compute_grid_features(grid_gdf,
                          road_gdf_dict=None,
                          landuse_gdf_dict=None,
                          city_crs=None,
                          max_workers=128):

    # STRtree Index
    road_indices = {}
    if road_gdf_dict:
        for rt, rd in road_gdf_dict.items():
            if rd is not None and not rd.empty:
                #rd = rd.to_crs(city_crs) if city_crs else rd
                road_gdf_dict[rt] = rd
                road_indices[rt] = STRtree(list(rd.geometry))
            else:
                road_indices[rt] = None

    landuse_indices = {}
    if landuse_gdf_dict:
        for lt, lu in landuse_gdf_dict.items():
            if lu is not None and not lu.empty:
                #lu = lu.to_crs(city_crs) if city_crs else lu
                landuse_gdf_dict[lt] = lu
                landuse_indices[lt] = STRtree(list(lu.geometry))
            else:
                landuse_indices[lt] = None

    def process_cell(idx, row):
        geom = row.geometry
        out = {'index': idx}

        # road length
        if road_gdf_dict:
            for rt, rd in road_gdf_dict.items():
                tree = road_indices[rt]
                if tree:
                    hits = tree.query(geom)
                    cand = rd.iloc[hits]
                    out[f"{rt}_length"] = cand.intersection(geom).length.sum() if not cand.empty else 0.0
                else:
                    out[f"{rt}_length"] = 0.0

        # land use area
        if landuse_gdf_dict:
            for lt, lu in landuse_gdf_dict.items():
                tree = landuse_indices[lt]
                if tree:
                    hits = tree.query(geom)
                    cand = lu.iloc[hits]
                    out[f"{lt}_area"] = cand.intersection(geom).area.sum() if not cand.empty else 0.0
                else:
                    out[f"{lt}_area"] = 0.0

        return out

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for cell_res in tqdm(
            executor.map(lambda row_tuple: process_cell(*row_tuple), grid_gdf.iterrows()),
            total=len(grid_gdf),
            desc="Computing grid features"
        ):
            results.append(cell_res)

    inds = pd.DataFrame(results).set_index('index')
    grid = grid_gdf.join(inds, how='left').fillna(0)

    grid['cellarea'] = grid.geometry.area

    # Compute landuse ratios & their sum
    #lu_cols = [c for c in grid.columns if c.endswith('_area')]
    lu_cols = [c for c in grid.columns 
           if c.endswith('_area') and c != 'water_area']
    #lu_cols = ["residential_area", "commercial_area", "industrial_area", "service_area", "parking_area", "recreational_area", "agriculture_area", "natural_area"]
    for col in lu_cols:
        grid[col.replace('_area', '_ratio')] = grid[col] / grid['cellarea']
    ratio_cols = [c for c in grid.columns 
                  if c.endswith('_ratio') and c != 'water_area_ratio']
    #ratio_cols = ["residential_ratio", "commercial_ratio", "industrial_ratio", "service_ratio", "parking_ratio", "recreational_ratio", "agriculture_ratio", "natural_ratio"]
    grid['lu_ratio_sum'] = grid[ratio_cols].sum(axis=1)

    grid['lu_exceeded'] = grid['lu_ratio_sum'] > 1
    mask = grid['lu_exceeded']
    for col in ratio_cols:
        grid.loc[mask, col] = grid.loc[mask, col] / grid.loc[mask, 'lu_ratio_sum']
    grid.loc[mask, 'lu_ratio_sum'] = 1.0

    # Compute total road length and road density (km per km²)
    rd_cols = [c for c in grid.columns if c.endswith('_length')]
    #rd_cols = ["rd_major_length", "rd_minor_length"]
    grid['total_road_length'] = grid[rd_cols].sum(axis=1)
    grid['road_density'] = (
        (grid['total_road_length'] / 1e3)  # km
        / (grid['cellarea'] / 1e6)        # km²
    )

    return grid


def check_data_files(data_dir, compute_cities, ghsl_uc_gdf):
    """
    For each city in compute_cities, check that the landuse, road, and building
    source files all exist. Report any missing files.
    """
    missing = {}

    for id_, city, country in zip(
        ghsl_uc_gdf["ID"],
        ghsl_uc_gdf["City"],
        ghsl_uc_gdf["Country"]
    ):
        if city not in compute_cities:
            continue

        # Paths to check
        landuse_path  = os.path.join(data_dir, 'osm_landuse_data',      f"landuse_{id_}.geojson")
        road_path     = os.path.join(data_dir, 'osm_road_service_data', f"road_service{id_}.shp")
        building_path = os.path.join(data_dir, 'osm_building_data',     f"building_{id_}.geojson")

        for name, path in [
            ("landuse",  landuse_path),
            ("road",     road_path),
            ("building", building_path)
        ]:
            if not os.path.exists(path):
                missing.setdefault(city, []).append((name, path))

    # Report results
    #if not missing:
        #print("✅ All required files are present for every city.")
    if missing:
        print("❌ Missing files detected:")
        for city, items in missing.items():
            print(f"  • {city}:")
            for name, path in items:
                print(f"      - missing {name}: {path}")



########################################################################################################################################################################

data_dir = "./meta_data/"
ghsl_uc_gdf = gpd.read_file(data_dir + "GHSL_UC_500samples.geojson") # .iloc[:2]
compute_cities = ghsl_uc_gdf['City'].to_list()
processed_cities = []

print(compute_cities)
check_data_files(data_dir, compute_cities, ghsl_uc_gdf)

total = len(ghsl_uc_gdf)
for idx, (id, city, country) in enumerate(zip(
        ghsl_uc_gdf["ID"],
        ghsl_uc_gdf["City"], 
        ghsl_uc_gdf["Country"]
    ), start=1):

    if city not in compute_cities:
        print(f"❌ {city} not in compute_cities!")
        continue

    boundary_gdf = ghsl_uc_gdf[(ghsl_uc_gdf['City'] == city) & (ghsl_uc_gdf['Country'] == country)].copy()
    if boundary_gdf.empty:
        print(f"❌ boundary_gdf of {city} is empty!")
        continue

    print(f"✅start , {city}, {id}, {idx}/{total}")
    city_crs = get_utm_crs(boundary_gdf)

    landuse_origin_geojson_path = os.path.join(data_dir, 'osm_landuse_data', f'landuse_{id}.geojson')
    landuse_converted_gpkg_path = os.path.join(data_dir, 'osm_landuse_data_converted', f'landuse_{id}.gpkg')
    if os.path.exists(landuse_origin_geojson_path):
        convert_geojson_to_gpkg(
            landuse_origin_geojson_path, #f"./data_collection/osm_landuse_data/landuse_{id}.geojson",
            landuse_converted_gpkg_path, #f"./data_collection/osm_landuse_data/landuse_{id}.gpkg",
            fields=["id", "fclass"])
    
    # Find road files
    road_path = os.path.join(data_dir, 'osm_road_service_data', f"road_service{id}.shp")
    if os.path.exists(road_path):
        road_gdf_dict = read_road_gdf(road_path, city_crs)
    else:
        road_gdf_dict = None
        print("❌ Cannot find road data!")

    # Find building andland use files
    building_path = os.path.join(data_dir, 'osm_building_data', f"building_{id}.geojson")
    landuse_path = os.path.join(data_dir, 'osm_landuse_data_converted', f"landuse_{id}.gpkg")
    if os.path.exists(landuse_path) and os.path.exists(building_path):
        landuse_gdf_dict = read_landuse_building_gdf(landuse_path, building_path, city_crs)
    elif os.path.exists(landuse_path):
        landuse_gdf_dict = read_landuse_gdf(landuse_path, city_crs)
        print("❌ Cannot find building data!")
    else:
        landuse_gdf_dict = None
        print("❌ Cannot find landuse data!")
        print("❌ Cannot find building data!")

    if landuse_gdf_dict is not None:
        landuse_gdf_dict = dedupe_landuse_dict(landuse_gdf_dict)

    # Find grid files
    city_dir = os.path.join("./meta_data/processed_data_500samples/", f"{city}/") 
    pattern = re.compile(r"^grid_r\d+_d\d+\_waterarea_density_gee.geojson$") # density_gee
    # print(os.listdir(city_dir))
    if os.path.exists(city_dir):
        grid_files = [f for f in os.listdir(city_dir) if (pattern.match(f) and f != 'grid_r0_d0.geojson')]
        print(grid_files)
    else:
        print(f"❌ Directory for {city} not found!")
        continue


    # Calculate
    start_time = time.time()
    for file in grid_files:
        #if os.path.exists(os.path.join(city_dir, file.split('.geojson')[0] + '_waterarea.geojson')): continue
        grid_gdf = gpd.read_file(os.path.join(city_dir, file)).to_crs(city_crs)
        #grid_gdf = grid_gdf.drop(columns=[col for col in grid_gdf.columns if col.startswith('built_')])
        #grid_gdf = grid_gdf.drop(columns=['water_area_ratio', 'population_count'])
        grid_gdf_with_features = compute_grid_features(
            grid_gdf,
            road_gdf_dict=road_gdf_dict,
            landuse_gdf_dict=landuse_gdf_dict,
            city_crs=city_crs,
            max_workers=128
        )

        original_columns = grid_gdf_with_features.columns.tolist()
        props_columns = [col for col in original_columns if col != 'geometry']

        # 1. Adding missing columns
        missing_columns = [col for col in standard_columns if col not in props_columns]
        for col in missing_columns:
            if col.endswith('_area') or col.endswith('_ratio'):
                grid_gdf_with_features[col] = 0.0
                print(f"📌 {city}/{file}: added missing column '{col}' (set to 0.0)")
            else:
                print(f"⚠️  {city}/{file}: missing non-_area/_ratio column '{col}' — skipped")

        # 2. Identify extra columns (non-standard)
        extra_columns = [col for col in props_columns if col not in standard_columns]
        if extra_columns:
            print(f"➕ {city}/{file}: extra columns appended at the end: {extra_columns}")

        # 3. Reordering
        new_column_order = [col for col in standard_columns if col in grid_gdf_with_features.columns] + \
                            [col for col in grid_gdf_with_features.columns if col not in standard_columns and col != 'geometry']

        grid_gdf_with_features = grid_gdf_with_features[new_column_order + ['geometry']]


        #grid_gdf_with_features.to_file(os.path.join(city_dir, file.split('.geojson')[0] + 'road_landuse.geojson'), driver="GeoJSON")
        base = file.split("_waterarea_density_gee.geojson")[0]
        out_name = f"{base}_density_landuse.geojson"
        grid_gdf_with_features.to_file(os.path.join(city_dir, out_name), driver="GeoJSON")

        #grid_waterarea_gdf = grid_waterarea_gdf.loc[grid_waterarea_gdf['water_area_ratio'] <= 0.5].copy()
        #grid_waterarea_gdf.to_file(os.path.join(city_dir, file.split('.geojson')[0] + '_waterarea.geojson'), driver="GeoJSON")
    print("✅done ", city, ", takes", time.time()-start_time, "s")