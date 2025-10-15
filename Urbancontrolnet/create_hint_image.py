import os
import time
import cv2
import numpy as np
import re
import math
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from concurrent.futures import ThreadPoolExecutor
from shapely.strtree import STRtree
from tqdm import tqdm

# 🔹 Compute UTM CRS for Projection
def get_utm_crs(boundary_gdf):
    centroid = boundary_gdf.to_crs("EPSG:4326").geometry.unary_union.centroid
    lat, lon = centroid.y, centroid.x
    utm_zone = math.floor((lon + 180) / 6) + 1
    epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
    return f"EPSG:{epsg_code}"

def read_water_gdf(water_path, projected_crs):
    water_in = gpd.read_file(water_path).to_crs(projected_crs)
    return water_in

def read_road_gdf(road_path, projected_crs):
    rd_gdf = gpd.read_file(road_path).to_crs(projected_crs)
    # print(rd_gdf.columns)

    # Define priority list for highway types
    priority_highways = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link']

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

    buffer_sizes = {
        'rd_m': 8.0,
        'rd_0': 8.0,
        'rd_1': 6.5,
        'rd_2': 5.0,
        'rd_3': 4.0,
    }

    level2fclass = {
        'rd_m': ['motorway', 'motorway_link'],
        'rd_0': ['trunk', 'trunk_link'],
        'rd_1': ['primary', 'primary_link'],
        'rd_2': ['secondary', 'secondary_link'],
        'rd_3': ['tertiary', 'tertiary_link'],
    }
    fclass2level = dict()
    for level in level2fclass:
        for fclass in level2fclass[level]:
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

    resolution = 32

    rd_list = dict()
    # Buffer the geometries
    for rd_type, buffer_size in buffer_sizes.items():
        rd_type_specific = get_group_as_gdf(rd_type)

        if rd_type_specific.empty:
            continue  # Skip if no geometries for this road type

        buffered_geometries = []
        # line_geometries = []
        for geom in rd_type_specific.geometry:
            buffered_geom = geom.buffer(distance=buffer_size, resolution=resolution, cap_style=2, join_style=1)
            if buffered_geom.geom_type == 'Polygon':
                buffered_geometries.append(buffered_geom)
            elif buffered_geom.geom_type == 'MultiPolygon':
                buffered_geometries.extend(list(buffered_geom.geoms))

        rd_list[rd_type] = gpd.GeoDataFrame(geometry=buffered_geometries, crs=rd_gdf.crs)

    return rd_list

def read_railway_gdf(railway_path, projected_crs):
    rl_vector = gpd.read_file(railway_path).to_crs(projected_crs)
    # Buffer parameters
    buffer_size = 0.7
    resolution = 32

    buffered_geometries = []

    for geom in rl_vector.geometry:
        buffered_geom = geom.buffer(distance=buffer_size, resolution=resolution, cap_style=2, join_style=1)
        if buffered_geom.geom_type == 'Polygon':
            buffered_geometries.append(buffered_geom)
        elif buffered_geom.geom_type == 'MultiPolygon':
            buffered_geometries.extend(list(buffered_geom.geoms))

    rl_buf = gpd.GeoDataFrame(geometry=buffered_geometries, crs=rl_vector.crs)
    return rl_buf.to_crs(projected_crs)

def rasterize_combined_area_with_water_classes(grid_gdf, output_folder, road_gdf_dict=None, railway_gdf=None, water_gdf=None, pixel_size=512, layer_styles=None):
    """
    Rasterizes road, railway, and water areas into TIFF images for each grid cell.
    - Uses parallel processing to speed up rasterization.
    """
    grid_gdf["water_area_ratio"] = 0.0  # Predefine column

    # ✅ Precompute spatial indices
    road_spatial_indices = {}
    if road_gdf_dict is not None:
        for road_type, road_gdf in road_gdf_dict.items():
            if road_gdf is not None and not road_gdf.empty:
                road_spatial_indices[road_type] = STRtree(list(road_gdf.geometry))
            else:
                road_spatial_indices[road_type] = None
    # road_spatial_indices = {road_type: STRtree(list(road_gdf.geometry)) for road_type, road_gdf in road_gdf_dict.items()}
    railway_spatial_index = STRtree(list(railway_gdf.geometry)) if railway_gdf is not None else None
    combined_spatial_index = STRtree(list(water_gdf.geometry)) if water_gdf is not None else None

    # ✅ **Multi-threading for faster processing**
    def process_grid_cell(idx, cell):
        cell_geometry = cell["geometry"]
        water_area_ratio = 0.0  # Default value
        cell_bounds = cell_geometry.bounds
        transform = from_bounds(*cell_bounds, pixel_size, pixel_size)

        out_image = np.ones((pixel_size, pixel_size, 3), dtype=np.uint8) * 255  # White background

        # 🌊 **Rasterize Water Layers**
        if combined_spatial_index:
            candidate_combined = water_gdf.iloc[[i for i in combined_spatial_index.query(cell_geometry)]]
            if not candidate_combined.empty:
                intersection = gpd.overlay(
                    gpd.GeoDataFrame({"geometry": [cell_geometry]}, crs=grid_gdf.crs),
                    candidate_combined,
                    how="intersection"
                )
                if not intersection.empty:
                    intersection['area'] = intersection.geometry.area
                    water_area_ratio = intersection.area.sum() / cell_geometry.area

                    shapes = [(geom, 1) for geom in intersection.geometry if not geom.is_empty]
                    rasterized_layer = rasterize(
                        shapes=shapes, out_shape=(pixel_size, pixel_size),
                        transform=transform, fill=0, all_touched=True, dtype=np.uint8
                    )
                    mask = rasterized_layer == 1
                    for i, color in enumerate(layer_styles['wa_in']['fill']):
                        out_image[:, :, i][mask] = color

        # 🛣 **Rasterize Road Layers**
        for road_type, spatial_index in reversed(list(road_spatial_indices.items())):
            candidate_roads = road_gdf_dict[road_type].iloc[[i for i in spatial_index.query(cell_geometry)]]
            if not candidate_roads.empty:
                intersection = gpd.overlay(
                    gpd.GeoDataFrame({"geometry": [cell_geometry]}, crs=grid_gdf.crs),
                    candidate_roads, how="intersection"
                )
                if not intersection.empty:
                    shapes = [(geom, 1) for geom in intersection.geometry if not geom.is_empty]
                    rasterized_layer = rasterize(
                        shapes=shapes, out_shape=(pixel_size, pixel_size),
                        transform=transform, fill=0, all_touched=True, dtype=np.uint8
                    )
                    mask = rasterized_layer == 1
                    for i, color in enumerate(layer_styles[road_type]['fill']):
                        out_image[:, :, i][mask] = color

        # 🚆 **Rasterize Railway Layers**
        if railway_gdf is not None and railway_spatial_index:
            candidate_railways = railway_gdf.iloc[[i for i in railway_spatial_index.query(cell_geometry)]]
            if not candidate_railways.empty:
                intersection = gpd.overlay(
                    gpd.GeoDataFrame({"geometry": [cell_geometry]}, crs=grid_gdf.crs),
                    candidate_railways, how="intersection"
                )
                if not intersection.empty:
                    shapes = [(geom, 1) for geom in intersection.geometry if not geom.is_empty]
                    rasterized_layer = rasterize(
                        shapes=shapes, out_shape=(pixel_size, pixel_size),
                        transform=transform, fill=0, all_touched=True, dtype=np.uint8
                    )
                    mask = rasterized_layer == 1
                    for i, color in enumerate(layer_styles['rl_buf']['fill']):
                        out_image[:, :, i][mask] = color

        # 🖼 **Save Output Image**
        file_name = f"{cell['row']}_{cell['col']}_{cell['r']}_{cell['d']}.tif"
        output_path = os.path.join(output_folder, file_name)
        with rasterio.open(output_path, "w", driver="GTiff", height=pixel_size, width=pixel_size,
                           count=3, dtype=out_image.dtype, crs=grid_gdf.crs, transform=transform,
                           compress="DEFLATE") as dst:
            for i in range(3):
                dst.write(out_image[:, :, i], i + 1)
        return idx, water_area_ratio

    # ✅ **Run Parallel Processing**
    results = {}
    with ThreadPoolExecutor(max_workers=128) as executor:
        for idx, water_ratio in tqdm(executor.map(lambda row_tuple: process_grid_cell(*row_tuple), list(grid_gdf.iterrows())), total=len(grid_gdf)):
            results[idx] = water_ratio  # Store in a dictionary
    grid_gdf["water_area_ratio"] = grid_gdf.index.map(results.get).fillna(0) 
    return grid_gdf



# **🔥 Run processing for all cities**
rd_color = (139, 0, 0) # (168,110,68)
layer_styles = {
    'rd_m':{'fill': rd_color, 'border': rd_color},
    'rd_0':{'fill': rd_color, 'border': rd_color},
    'rd_1':{'fill': rd_color, 'border': rd_color}, # 68, 4, 87
    'rd_2':{'fill': rd_color, 'border': rd_color}, # 68, 4, 87
    'rd_3':{'fill': rd_color, 'border': rd_color}, # 68, 4, 87
    'wa_in': {'fill': (70, 130, 180), 'border': (70, 130, 180)},
    'rl_buf': {'fill': (30,0,5), 'border': (30,0,5)},
}
data_dir = "./urban_data/meta_data/"
osm_data_dir = data_dir + "osm_data/"
osm_road_data_dir = data_dir + "osm_road_service_data/"
ghsl_uc_gdf = gpd.read_file(data_dir + "GHSL_UC_500samples.geojson") # .iloc[:2]


for id, city, country in zip(ghsl_uc_gdf["ID"], ghsl_uc_gdf["City"], ghsl_uc_gdf["Country"]):
    # if city in computed_cities:
    #     continue

    # Find grid files
    city_dir = os.path.join(data_dir, f"meta_data_500samples/{city}/")
    pattern = re.compile(r"^grid_r\d+_d\d+_waterarea\.geojson$") # density_gee
    # print(os.listdir(city_dir))
    # grid_files = [f for f in os.listdir(city_dir) if (pattern.match(f) and f != 'grid_r0_d0.geojson')]
    grid_files = [f for f in os.listdir(city_dir) if pattern.match(f)]
    # print(grid_files)
    # grid_files = ['grid_r0_d0.geojson']
    city_hint_dir = os.path.join(data_dir, f"processed_data_500samples/{city}/hint_image/")
    os.makedirs(city_hint_dir, exist_ok=True)
    start_time = time.time()
    # if os.path.exists(os.path.join(city_dir, 'grid_r0_d0_waterarea.geojson')): continue

    boundary_gdf = ghsl_uc_gdf[(ghsl_uc_gdf['City'] == city) & (ghsl_uc_gdf['Country'] == country)].copy()
    # print(boundary_gdf)
    if boundary_gdf.empty:
        continue

    print("✅start ", city)
    city_crs = get_utm_crs(boundary_gdf)
    if os.path.exists(osm_data_dir + f"water_{id}.geojson"):
        water_gdf = read_water_gdf(osm_data_dir + f"water_{id}.geojson", city_crs)
    else:
        water_gdf = None
    # print(water_gdf)
    # print(os.path.exists(osm_data_dir + f"road_{id}.shp"), osm_data_dir + f"road_{id}.shp")
    if os.path.exists(osm_road_data_dir + f"road_service{id}.shp"):
        road_gdf_dict = read_road_gdf(osm_road_data_dir + f"road_service{id}.shp", city_crs)
    else:
        road_gdf_dict = None
    # print(road_gdf_dict)
    if os.path.exists(osm_data_dir + f"railway_{id}.geojson"):
        railway_gdf = read_railway_gdf(osm_data_dir + f"railway_{id}.geojson", city_crs)
    else:
        railway_gdf = None
    # print(railway_gdf)
    
    for file in grid_files:
        if os.path.exists(os.path.join(city_dir, file.split('.geojson')[0] + '_waterarea2.geojson')): continue
        grid_gdf = gpd.read_file(os.path.join(city_dir, file)).to_crs(city_crs)
        grid_waterarea_gdf = rasterize_combined_area_with_water_classes(grid_gdf, city_hint_dir, road_gdf_dict, railway_gdf, water_gdf, pixel_size=512, layer_styles=layer_styles)
        grid_waterarea_gdf = grid_waterarea_gdf.loc[grid_waterarea_gdf['water_area_ratio'] <= 0.5].copy()
        grid_waterarea_gdf.to_file(os.path.join(city_dir, file.split('.geojson')[0] + '_waterarea2.geojson'), driver="GeoJSON")
        # grid_waterarea_gdf.to_file(os.path.join(city_dir, file.split('_gee.geojson')[0] + '_waterarea.geojson'), driver="GeoJSON")
    print("✅done ", city, ", takes", time.time()-start_time, "s")