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
from rasterio.mask import mask
from shapely.geometry import box

# 🔹 Compute UTM CRS for Projection
def get_utm_crs(boundary_gdf):
    centroid = boundary_gdf.to_crs("EPSG:4326").geometry.unary_union.centroid
    lat, lon = centroid.y, centroid.x
    utm_zone = math.floor((lon + 180) / 6) + 1
    epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
    return f"EPSG:{epsg_code}"


# Reproject DEM to match grid CRS
def reproject_raster(src_path, dst_path, dst_crs):
    with rasterio.open(src_path) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=rasterio.enums.Resampling.nearest)
    
    return dst_path  # Return new raster file path

def crop_geotiff_by_grid(geotiff_path, geotiff_tmp_path, grid_gdf, city_crs, output_dir, buffer_distance=30):
    """
    Crops a GeoTIFF based on each row in grid_gdf after buffering by 30m, 
    crops 1-pixel edges, and saves each as an individual GeoTIFF.

    Parameters:
    - geotiff_path (str): Path to the input DEM GeoTIFF.
    - grid_gdf (GeoDataFrame): GeoDataFrame with each row as a polygon (400m x 400m grid).
    - output_dir (str): Directory to save cropped GeoTIFFs.
    - buffer_distance (int, optional): Buffer size in meters. Default is 30m.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    grid_gdf = grid_gdf.to_crs(city_crs)
    # Open the main GeoTIFF file
    tmp_geotiff_path = reproject_raster(geotiff_path, geotiff_tmp_path, city_crs)
    with rasterio.open(tmp_geotiff_path) as src:

        pixel_size_x, pixel_size_y = src.res  # Get pixel resolution
        print("pixel_size_x", pixel_size_x, "pixel_size_y", pixel_size_y)
        nodata_value = src.nodata  # Get NoData value

        # Iterate over each grid cell
        for idx, row in tqdm(grid_gdf.iterrows(), total=len(grid_gdf), desc="Cropping DEM"):
            try:
                # Step 1: Buffer the grid cell
                buffered_geom = row.geometry.buffer(buffer_distance)
                buffered_geom = gpd.GeoSeries(buffered_geom, crs=grid_gdf.crs).iloc[0]
                # Step 2: Get the bounding box of the buffered grid
                minx, miny, maxx, maxy = buffered_geom.bounds
                bbox_geom = box(minx, miny, maxx, maxy)

                # Step 3: Crop raster using the buffered bounding box
                out_image, out_transform = mask(src, [bbox_geom], crop=True, nodata=nodata_value)
                out_meta = src.meta.copy()

                # Step 4: Crop 1-pixel from the edges
                if out_image.shape[1] > 2 and out_image.shape[2] > 2:
                    out_image = out_image[:, 1:-1, 1:-1]  # Remove 1 pixel from all sides
                # print(out_image.shape)
                # Update metadata
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                # Generate file name based on grid attributes
                row_value = row["row"]
                col_value = row["col"]
                r_value = row["r"]
                d_value = row["d"]
                file_name = f"{row_value}_{col_value}_{r_value}_{d_value}.tif"
                output_filename = os.path.join(output_dir, file_name)

                # Save cropped DEM
                with rasterio.open(output_filename, "w", **out_meta) as dest:
                    dest.write(out_image)

                # print(f"✅ Saved: {output_filename}")

            except Exception as e:
                print(f"⚠️ Skipping grid {idx} due to error: {e}")


# **🔥 Run processing for all cities**
data_dir = "./urban_data/meta_data/"
dem_data_dir = data_dir + "FABDEM_geotifs/"
ghsl_uc_gdf = gpd.read_file(data_dir + "GHSL_UC_500samples.geojson")


for id, city, country in zip(ghsl_uc_gdf["ID"], ghsl_uc_gdf["City"], ghsl_uc_gdf["Country"]):
    # if city not in ['Dakar']:
    #     continue

    boundary_gdf = ghsl_uc_gdf[(ghsl_uc_gdf['City'] == city) & (ghsl_uc_gdf['Country'] == country)].copy()
    # print(boundary_gdf)
    if boundary_gdf.empty:
        continue

    city_crs = get_utm_crs(boundary_gdf)  

    city_dem_dir = os.path.join(data_dir, f"processed_data_500samples/{city}/dem_image/")
    os.makedirs(city_dem_dir, exist_ok=True)
    city_geotiff_path = os.path.join(dem_data_dir, f"FABDEM_{id}.tif")

    # with rasterio.open(city_geotiff_path) as src:
    #     crs = src.crs
    #     if crs != 'EPSG:4326': print("CRS:", crs)
    city_tmp_geotiff_path = os.path.join(dem_data_dir, f"FABDEM_tmp_{id}.tif")
    # Find grid files
    city_dir = os.path.join(data_dir, f"meta_data_500samples/{city}/")
    pattern = re.compile(r"^grid_r\d+_d\d+_waterarea\.geojson$")
    grid_files = [f for f in os.listdir(city_dir) if pattern.match(f)]
    start_time = time.time()
    for file in grid_files:
        grid_gdf = gpd.read_file(os.path.join(city_dir, file))#.to_crs(4326)
        # geotiff_path, grid_gdf, output_dir
        crop_geotiff_by_grid(city_geotiff_path, city_tmp_geotiff_path, grid_gdf, city_crs, city_dem_dir)
    print("✅done ", city, ", takes", time.time()-start_time, "s")
