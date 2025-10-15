import os
import cv2
import numpy as np
import re
import math
import geopandas as gpd
from shapely.geometry import box
from concurrent.futures import ThreadPoolExecutor  # ✅ Multi-threading for faster tile processing


# 🔹 Compute UTM CRS for Projection
def get_utm_crs(boundary_gdf):
    centroid = boundary_gdf.to_crs("EPSG:4326").geometry.unary_union.centroid
    lat, lon = centroid.y, centroid.x
    utm_zone = math.floor((lon + 180) / 6) + 1
    epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
    return f"EPSG:{epsg_code}"



def crop_boundary(boundary_gdf, city_crs):
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

    # Define a 30 km × 30 km bounding box
    minx, maxx = centroid_x - 12500, centroid_x + 12500
    miny, maxy = centroid_y - 12500, centroid_y + 12500

    # Create a bounding box geometry
    bounding_box = box(minx, miny, maxx, maxy)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bounding_box]}, crs=city_crs)

    # Perform spatial intersection (clipping)
    cropped_boundary = gpd.overlay(boundary_gdf_meters, bbox_gdf, how='intersection')

    # Plot the result (optional)
    cropped_boundary.plot()

    return cropped_boundary



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

        # print(f"Zoom Level: {zoom}")
        # print(f"Tile Width in meters: {width_meters}")
        # print(f"Tile Height in meters: {height_meters}")

        # Check if the tile size is close to the target distance
        if (width_meters + height_meters) / 2 <= distance:
            return zoom

    return 17  # Default to highest zoom if no match is found

# Function to load a tile image
def load_tile_image(zoom, x, y, tile_folder):
    filepath = os.path.join(tile_folder, f"{zoom}_{x}_{y}.png")
    return cv2.imread(filepath) if os.path.exists(filepath) else None

# Function to merge multiple tiles into one image
def merge_tiles(tiles, zoom, tiles_base_path):
    images = []
    tile_width, tile_height = None, None

    # Load all tile images in one go to avoid redundant I/O
    for xtile, ytile in tiles:
        img = load_tile_image(zoom, xtile, ytile, tiles_base_path)
        if img is None:
            print(f"❌ Missing tile {xtile}, {ytile}. Skipping...")
            return None
        images.append((img, xtile, ytile))
        if tile_width is None:
            tile_height, tile_width, _ = img.shape

    if not images:
        return None

    # Compute grid size
    min_xtile, max_xtile = min(t[1] for t in images), max(t[1] for t in images)
    min_ytile, max_ytile = min(t[2] for t in images), max(t[2] for t in images)

    total_width = (max_xtile - min_xtile + 1) * tile_width
    total_height = (max_ytile - min_ytile + 1) * tile_height

    # Create merged image
    merged_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    # Place tiles onto the canvas
    for img, xtile, ytile in images:
        x_offset = (xtile - min_xtile) * tile_width
        y_offset = (ytile - min_ytile) * tile_height
        merged_image[y_offset:y_offset + tile_height, x_offset:x_offset + tile_width] = img

    return merged_image

# Function to crop image based on geographic bounding box
def crop_and_resize_image(image_A, A_left, A_right, A_lower, A_upper, B_left, B_right, B_lower, B_upper, output_size=(512, 512)):
    A_height, A_width, _ = image_A.shape

    # Convert geographic bounding box to pixel coordinates
    def latlon_to_pixel(lat, lon):
        x_pixel = A_width * (lon - A_left) / (A_right - A_left)
        y_pixel = A_height * (A_upper - lat) / (A_upper - A_lower)
        return int(x_pixel), int(y_pixel)

    left_pixel, upper_pixel = latlon_to_pixel(B_upper, B_left)
    right_pixel, lower_pixel = latlon_to_pixel(B_lower, B_right)

    # Ensure pixel coordinates are within the image bounds
    left_pixel, right_pixel = max(left_pixel, 0), min(right_pixel, A_width)
    upper_pixel, lower_pixel = max(upper_pixel, 0), min(lower_pixel, A_height)

    # Crop and resize image
    cropped_image = image_A[upper_pixel:lower_pixel, left_pixel:right_pixel]
    return cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_LINEAR)

# Function to process a single grid cell in parallel
def process_grid_cell(grid_row, zoom_level, tiles_base_path, city_satellite_dir):
    polygon = grid_row['geometry']
    output_path = os.path.join(city_satellite_dir, f"{grid_row['row']}_{grid_row['col']}_{grid_row['r']}_{grid_row['d']}.jpg")

    if os.path.exists(output_path):
        return  # Skip already processed files

    # Function to convert tile indices to geographic coordinates (WGS 84)
    def tile_to_deg(xtile, ytile, zoom):
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = lat_rad * (180.0 / math.pi)
        return lat_deg, lon_deg

    # Function to convert geographic coordinates to tile indices
    def deg_to_tile(lat_deg, lon_deg, zoom):
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat_deg)
        ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        return xtile, ytile

    # Compute tile bounds
    minx, miny, maxx, maxy = polygon.bounds
    min_tile_x, min_tile_y = deg_to_tile(miny, minx, zoom_level)
    max_tile_x, max_tile_y = deg_to_tile(maxy, maxx, zoom_level)

    min_tile_x, max_tile_x = min(min_tile_x, max_tile_x), max(min_tile_x, max_tile_x)
    min_tile_y, max_tile_y = min(min_tile_y, max_tile_y), max(min_tile_y, max_tile_y)
    tiles = [(x, y) for x in range(min_tile_x, max_tile_x + 1) for y in range(min_tile_y, max_tile_y + 1)]

    # Merge tiles
    merged_image = merge_tiles(tiles, zoom_level, tiles_base_path)
    if merged_image is None:
        print("merged_image is None", output_path)
        return

    # Compute geographic bounds of the merged image
    A_upper, A_left = tile_to_deg(min_tile_x, min_tile_y, zoom_level)
    A_lower, A_right = tile_to_deg(max_tile_x + 1, max_tile_y + 1, zoom_level)

    # Crop and save image
    cropped_image = crop_and_resize_image(merged_image, A_left, A_right, A_lower, A_upper, minx, maxx, miny, maxy)

    def save_image_high_quality(image, output_path, format='jpeg'):
        """
        Save the image to a file in the highest possible quality.

        Parameters:
        - image: The image to save (NumPy array from OpenCV).
        - output_path: File path where the image will be saved.
        - format: Format to save the image ('jpeg' or 'png').
        """
        # Check format and set the file extension
        if format.lower() == 'jpeg':
            output_path = output_path if output_path.endswith('.jpg') else output_path + '.jpg'
            # Save with maximum quality
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        elif format.lower() == 'png':
            output_path = output_path if output_path.endswith('.png') else output_path + '.png'
            # Save with no compression
            cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            raise ValueError("Unsupported format. Use 'jpeg' or 'png'.")
    
    save_image_high_quality(cropped_image, output_path, format='jpeg')

# Main function to process all cities
def process_city_images(city, country, ghsl_uc_gdf, zoom_level, data_dir):
    city_dir = os.path.join(data_dir, f"meta_data_500samples/{city}/")
    tiles_base_path = os.path.join(data_dir, f"mapbox_tiles/{city}")
    city_satellite_dir = os.path.join(data_dir, f"processed_data_500samples/{city}/satellite_image/")
    os.makedirs(city_satellite_dir, exist_ok=True)

    # Find grid files
    pattern = re.compile(r"^grid_r\d+_d\d+_waterarea\.geojson$")
    grid_files = [f for f in os.listdir(city_dir) if pattern.match(f)]
    print(grid_files)
    for file in grid_files:
        grid_gdf = gpd.read_file(os.path.join(city_dir, file)).to_crs(4326)
        # **🔥 Multi-threading for faster processing**
        with ThreadPoolExecutor(max_workers=80) as executor:
            executor.map(lambda row_tuple: process_grid_cell(row_tuple[1], zoom_level, tiles_base_path, city_satellite_dir), grid_gdf.iterrows())

    print(f"✅ {city} processed: {len(os.listdir(city_satellite_dir))} images saved.")

# **🔥 Run processing for all cities**
size = (512, 512)
data_dir = "./urban_data/meta_data/"
ghsl_uc_gdf = gpd.read_file(data_dir + "GHSL_UC_500samples.geojson")
# computed_cities = ["Singapore", "Hong Kong", "Kigali", "Kinshasa", "Mexico City", "São Paulo", "Orlando", "Chicago", "Munich", "Stockholm"]

# Tokyo, Chennai, 
for city, country in zip(ghsl_uc_gdf["City"], ghsl_uc_gdf["Country"]):
    # if city in ["Dakar"]:
    #     continue

    boundary_gdf = ghsl_uc_gdf[(ghsl_uc_gdf['City'] == city) & (ghsl_uc_gdf['Country'] == country)].copy()
    if boundary_gdf.empty:
        continue

    city_crs = get_utm_crs(boundary_gdf)
    boundary_gdf = crop_boundary(boundary_gdf, city_crs)
    zoom_level = define_zoom(boundary_gdf, city_crs)

    process_city_images(city, country, ghsl_uc_gdf, zoom_level, data_dir)

    # break
