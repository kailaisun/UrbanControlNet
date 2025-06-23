import json
import os
import csv
import random
# Munich文件夹的路径，包含GeoJSON文件和图片目录
# GenAI_density文件夹的路径
base_path = './urban_data/tencities/GenAI_density/'

# 获取所有城市的目录
cities = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

def check_files_existence(directory,directory_hint, filenames,city):
    existing_files = os.listdir(directory)
    existing_files_hint=os.listdir(directory_hint)
    count=0
    results = []
    for filename, filename_hint,feature in filenames:
        if filename in existing_files:
            if filename_hint in existing_files_hint:
                count = count + 1
                # 提取、处理并打印建筑数据
                built_volume_total = feature['properties'].get('built_volume_total', 0)
                built_volume_total = round(built_volume_total / 1000, 1)  # 转换为千立方米

                built_volume_nres = round(feature['properties'].get('built_volume_nres', 0),1)


                built_surface_total = feature['properties'].get('built_surface_total', 0)
                built_surface_total = round(built_surface_total / 1000, 1)  # 转换为千平方米

                built_surface_nres = round(feature['properties'].get('built_surface_nres', 0),1)

                built_height_mean=round(feature['properties'].get('built_height_mean', 0),1)
                built_height_std=round(feature['properties'].get('built_height_std', 0),1)

                population_count = feature['properties'].get('population_count', 0)
                population_count = round(population_count, 1)


                results.append({
                    'city': city,
                    'built_volume_total': round(built_volume_total, 1),
                    'built_volume_nres': round(built_volume_nres, 1),
                    'built_surface_total': round(built_surface_total, 1),
                    'built_surface_nres': round(built_surface_nres, 1),
                    'built_height_mean': round(built_height_mean, 1),
                    'built_height_std': round(built_height_std, 1),
                    'population_count': round(population_count, 1),
                })
            else:
                print(f"{filename_hint}: 不存在")
        else:
            print(f"{filename}: 不存在")
    return results

# 检查每个GeoJSON文件
all_results = []
for city in cities:
    city_path = os.path.join(base_path, city)
    print(city_path)
    geojson_files = [os.path.join(city_path, f) for f in os.listdir(city_path) if f.endswith('.geojson')]

    total_count = 0
    for geojson_path in geojson_files:
        # 从文件名提取r和d的值，构建正确的图片目录名
        filename_parts = geojson_path.split('/')[-1].replace('.geojson', '').split('_')
        r_value = filename_parts[2]  # 提取r的值
        d_value = filename_parts[3]  # 提取d的值
        directory_name = f"{r_value}_{d_value}_cv"
        directory_path = os.path.join(city_path, 'satellite_images_z17', directory_name)
        directory_name_hint = f"{r_value}_{d_value}"
        directory_path_hint = os.path.join(city_path, 'hint_images', directory_name_hint)

        # 确保目录存在
        if not os.path.exists(directory_path):
            print(f"目录 {directory_path} 不存在")
            continue

        if not os.path.exists(directory_path_hint):
            print(f"目录 {directory_path_hint} 不存在")
            continue

        # 加载GeoJSON数据
        with open(geojson_path, 'r') as file:
            geojson_data = json.load(file)

        # 生成图片文件名
        image_filenames = [
            (f"{feature['properties']['row']}_{feature['properties']['col']}_{directory_name}.jpg",f"{feature['properties']['row']}_{feature['properties']['col']}_{directory_name_hint}.tif", feature)
            for feature in geojson_data['features']
        ]



        result=check_files_existence(directory_path, directory_path_hint,image_filenames, city)
        total_count+=len(result)
        all_results.extend(result)

    print(f"Total files found for {city}: {total_count}")

    # 保存结果到 CSV 文件
csv_file_path = './urban_data/tencities/results.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['city', 'built_volume_total', 'built_volume_nres','built_surface_total','built_surface_nres' ,'built_height_mean','built_height_std','population_count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(all_results)

print(f"Results saved to {csv_file_path}")


