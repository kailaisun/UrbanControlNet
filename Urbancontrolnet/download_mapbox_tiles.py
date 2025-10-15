import os
import geopandas as gpd
import requests
from PIL import Image
from io import BytesIO
import math
from shapely.geometry import box
import concurrent.futures
from itertools import cycle
import cv2

# 🔹 Mapbox API Credentials
TILESET_ID = "mapbox.satellite"
TILE_URL_TEMPLATE = "https://api.mapbox.com/v4/{tileset}/{z}/{x}/{y}@2x.png?access_token={token}"
MAPBOX_TOKENS = [ ]
TOKEN_CYCLE = cycle(MAPBOX_TOKENS)  # Rotates tokens automatically
# 🔹 Define Output Directory
DATA_DIR = "./urban_data/meta_data/"

# 🔹 Max Concurrent Downloads
MAX_WORKERS = 64  # Adjust based on server capacity
MAX_RETRIES = 3  # Number of retries on failure

# 🔹 Compute UTM CRS for Projection
def get_utm_crs(boundary_gdf):
    centroid = boundary_gdf.to_crs("EPSG:4326").geometry.unary_union.centroid
    lat, lon = centroid.y, centroid.x
    utm_zone = math.floor((lon + 180) / 6) + 1
    epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
    return f"EPSG:{epsg_code}"


def crop_buffer_boundary(boundary_gdf, city_crs, buffer_distance=400):
    # If the area is small (< 400 km²), return directly with correct CRS
    if boundary_gdf['Area_Km2'].tolist()[0] < 500:
        return boundary_gdf.to_crs(city_crs)

    # Convert to the projected CRS (meters)
    if boundary_gdf.crs != city_crs:
        boundary_gdf_meters = boundary_gdf.to_crs(city_crs)
    else:
        boundary_gdf_meters = boundary_gdf.copy()

    # Get the centroid of the entire boundary
    centroid = boundary_gdf_meters.geometry.unary_union.centroid
    centroid_x, centroid_y = centroid.x, centroid.y

    # Define a 20 km × 20 km bounding box
    minx, maxx = centroid_x - 12500 - buffer_distance, centroid_x + 12500 + buffer_distance
    miny, maxy = centroid_y - 12500 - buffer_distance, centroid_y + 12500 + buffer_distance

    # Create a bounding box geometry
    bounding_box = box(minx, miny, maxx, maxy)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bounding_box]}, crs=city_crs)

    # Perform spatial intersection (clipping)
    cropped_boundary = gpd.overlay(boundary_gdf_meters, bbox_gdf, how='intersection')

    # # Plot the result (optional)
    # cropped_boundary.plot()

    return cropped_boundary

# 🔹 Convert lat/lon to tile coordinates
def deg_to_tile(lat_deg, lon_deg, zoom):
    lat_rad = lat_deg * (math.pi / 180.0)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad)), math.e) / math.pi) / 2.0 * n)
    return (xtile, ytile)

# 🔹 Convert tile numbers to bounding box
def tile_to_bbox(x, y, zoom):
    n = 2.0 ** zoom
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return box(lon_min, lat_min, lon_max, lat_max)

def define_zoom(boundary_gdf, city_crs, distance=400):
    """
    Determines the best zoom level where the tile size is close to `distance` meters.

    Parameters:
    - boundary_gdf (GeoDataFrame): City boundary in any CRS.
    - distance (float): Target distance in meters for tile size.

    Returns:
    - int: Optimal zoom level.
    """
    # Get centroid in lat/lon (EPSG:4326)
    centroid = boundary_gdf.to_crs("EPSG:4326").geometry.unary_union.centroid
    lat, lon = centroid.y, centroid.x
    # utm_crs = get_utm_crs(lat, lon)
    # Iterate through zoom levels to find the best fit
    for zoom in [16, 17]:
        n = 2.0 ** zoom  # Number of tiles at this zoom level

        # Convert lon/lat to Web Mercator (EPSG:3857) tile coordinates
        tile_x = (lon + 180.0) / 360.0 * n
        tile_y = (1 - math.log(math.tan(math.radians(lat)) + (1 / math.cos(math.radians(lat)))) / math.pi) / 2.0 * n

        # Compute tile boundaries in lon/lat
        lon_min = (tile_x / n) * 360.0 - 180.0
        lon_max = ((tile_x + 1) / n) * 360.0 - 180.0
        lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (tile_y + 1) / n))))
        lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n))))

        # Create bounding box in lat/lon
        bbox = box(lon_min, lat_min, lon_max, lat_max)
        gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs="EPSG:4326")

        # Convert to projected CRS for accurate meter-based calculations
        projected_gdf = gdf.to_crs(city_crs)
        projected_bounds = projected_gdf.total_bounds  # [minx, miny, maxx, maxy]
        
        # Compute width and height in meters
        width_meters = projected_bounds[2] - projected_bounds[0]
        height_meters = projected_bounds[3] - projected_bounds[1]

        print(f"Zoom Level: {zoom}")
        print(f"Tile Width in meters: {width_meters}")
        print(f"Tile Height in meters: {height_meters}")

        # Check if the tile size is close to the target distance
        if (width_meters + height_meters) / 2 <= distance:
            return zoom

    return 17  # Default to highest zoom if no match is found

# 🔹 Download Tile Function (Thread-Safe)
def download_tile(session, x, y, zoom, output_dir, token):
    tile_path = os.path.join(output_dir, f"{zoom}_{x}_{y}.png")
    if os.path.exists(tile_path) and cv2.imread(tile_path):
        return

    url = TILE_URL_TEMPLATE.format(tileset=TILESET_ID, z=zoom, x=x, y=y, token=token)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            img.save(tile_path)
            return  # Successful download, exit function
        except requests.exceptions.RequestException as e:
            print(f"❌ Attempt {attempt+1} failed for {x}, {y}, {zoom}: {e}")
    
    print(f"🚨 Failed to download {x}, {y}, {zoom} after {MAX_RETRIES} attempts.")

# 🔹 Process Each City Efficiently with Correct Token Rotation
def process_city(id, city, country, ghsl_uc_gdf):
    boundary_gdf = ghsl_uc_gdf.loc[(ghsl_uc_gdf['ID'] == id)].copy()
    if boundary_gdf.empty:
        return
    city_crs = get_utm_crs(boundary_gdf)
    boundary_gdf = crop_buffer_boundary(boundary_gdf, city_crs)
    zoom_level = define_zoom(boundary_gdf, city_crs, distance=400)  # Adjust as needed

    # 🔹 Compute Tile Range
    boundary_wgs84 = boundary_gdf.to_crs(4326)
    min_lon, min_lat, max_lon, max_lat = boundary_wgs84.total_bounds
    min_tile = deg_to_tile(min_lat, min_lon, zoom_level)
    max_tile = deg_to_tile(max_lat, max_lon, zoom_level)

    output_dir = os.path.join(DATA_DIR, f"mapbox_tiles/{city}")
    os.makedirs(output_dir, exist_ok=True)

    # 🔹 Generate List of Tiles with Pre-Assigned Tokens
    tile_list = [
        (x, y, next(TOKEN_CYCLE))  # Pre-assign a token
        for x in range(min_tile[0], max_tile[0] + 1)
        for y in range(max_tile[1], min_tile[1] + 1)
        if boundary_wgs84.intersects(tile_to_bbox(x, y, zoom_level)).any()
    ]

    print(f"📍 {city}: {len(tile_list)} tiles to download.")

    # 🔹 Parallel Download Using ThreadPoolExecutor
    with requests.Session() as session:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(lambda tile: download_tile(session, tile[0], tile[1], zoom_level, output_dir, tile[2]), tile_list)

# 🔹 Load Cities and Start Processing
# df = gpd.read_file(os.path.join(DATA_DIR, "GHSL_UC_above100_matched.geojson"))
# df = df.loc[df['Area_Km2'] >= 200].copy()
# computed_df = gpd.read_file(os.path.join(DATA_DIR, "GHSL_UC_100samples.geojson"))
# df = df.loc[~(df['ID'].isin(computed_df['ID'].tolist()))].copy()
# sample_df = df.sample(n=200, replace=False)
# sample_df.to_file(os.path.join(DATA_DIR, "GHSL_UC_200samples_v2.geojson"), driver="GeoJSON")
# print(len(sample_df))

ghsl_uc_gdf = gpd.read_file(os.path.join(DATA_DIR, "GHSL_UC_500samples.geojson"))

total_tiles_downloaded = 0

for id, city, country in zip(ghsl_uc_gdf["ID"].tolist(), ghsl_uc_gdf["City"].tolist(), ghsl_uc_gdf["Country"].tolist()):
    process_city(id, city, country, ghsl_uc_gdf)

