import json
import os
import numpy as np
import random
# Munich文件夹的路径，包含GeoJSON文件和图片目录
# GenAI_density文件夹的路径
base_path = './urban_data/meta_data/processed_data_388samples'

# 获取所有城市的目录
cities = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

n1=0
n2=0
n3=0
n4=0
def check_files_existence(directory,directory_hint,directory_dem, filenames,city):
    global n1,n2,n3,n4
    existing_files = os.listdir(directory)
    existing_files_hint=os.listdir(directory_hint)
    existing_files_dem=os.listdir(directory_dem)
    count=0
    RVP_count=0
    results = []
    count_incomplete = 0
    count_complete = 0
    for filename, filename_hint,filename_dem,feature in filenames:
        if filename in existing_files:
            if filename_hint in existing_files_hint and filename_dem in existing_files_dem:
                water = feature['properties'].get('water_area_ratio', 0)
                water = round(water, 1)
                if water >=0.5:
                    continue

                count = count + 1
                # 提取、处理并打印建筑数据
                built_volume_total = feature['properties'].get('built_volume_total', 0)
                built_volume_total = round(built_volume_total / 1000, 1)  # 转换为千立方米
                built_surface_total = feature['properties'].get('built_surface_total', 0)
                built_surface_total = round(built_surface_total / 1000, 1)  # 转换为千平方米
                built_surface_nres= feature['properties'].get('built_surface_nres', 0)
                built_surface_nres = round(built_surface_nres / 1000, 1)
                population_count = feature['properties'].get('population_count', 0)
                population_count = round(population_count, 1)
                built_volume_nres=feature['properties'].get('built_volume_nres', 0)
                built_volume_nres = round(built_volume_nres / 1000, 1)
                road_density=feature['properties'].get('road_density', 0)
                road_density = round(road_density , 1)

                # Four
                # metrics: (1)
                # Residential
                # Population
                # Density(RPD), (2)
                # Residential
                # Volume
                # Per
                # Capita(RVP), (3)
                # Building
                # Volume
                # Density(BVD=Building
                # Volumes / Land
                # areas), and (4)
                # Building
                # Coverage
                # Ratio(BCR=Building
                # areas / land
                # # areas).
                land = (1 - water) * 40 * 4  # 转换为千平方米

                if built_surface_total > built_surface_nres:
                    Residential_Population_Density = round(
                        population_count / (built_surface_total - built_surface_nres), 2)
                else:
                    Residential_Population_Density = 0
                if population_count>0:
                    Residential_Volume_Per_Capita=round((built_volume_total-built_volume_nres)/population_count*1000,2)
                else:
                    Residential_Volume_Per_Capita=0
                if Residential_Volume_Per_Capita>1000:
                    RVP_count=RVP_count + 1
                    Residential_Volume_Per_Capita=1000


                # Building_Coverage_Ratio=round(built_surface_total/land*100,2)
                Building_Coverage_Ratio = round(built_surface_total/160 * 100, 2)
                if Building_Coverage_Ratio > 100:
                    print(1)
                    continue
                Building_Volume_Density=round(built_volume_total/160,2)



                lu_ratio_sum=feature['properties'].get('lu_ratio_sum', 0)
                if lu_ratio_sum <= 0.5:
                    count_incomplete=count_incomplete+1
                    prompt = (f"Satellite imagery of {city}. "
                          f"The Building Coverage Ratio in this area is {Building_Coverage_Ratio} %."
                          f"The Building Volume Density is {Building_Volume_Density} cubic meters per square meter."
                          f"The Road Density is {road_density} kilometers per square kilometer.")
                else:
                    count_complete=count_complete+1
                    properties = feature['properties']

                    # 获取各类比例（可按需扩展）
                    residential_ratio = properties.get('residential_ratio', 0)
                    commercial_ratio = properties.get('commercial_ratio', 0)
                    industrial_ratio = properties.get('industrial_ratio', 0)
                    parking_ratio = properties.get('parking_ratio', 0)
                    recreational_ratio = properties.get('recreational_ratio', 0)
                    natural_ratio = properties.get('natural_ratio', 0)

                    # 构建 land use 描述列表（自动过滤为 0 的）
                    landuse_descriptions = []

                    if residential_ratio > 0 and round(residential_ratio * 100) > 0:
                        landuse_descriptions.append(f"{residential_ratio * 100:.0f} percent of residential")
                    if commercial_ratio > 0 and round(commercial_ratio * 100) > 0:
                        landuse_descriptions.append(f"{commercial_ratio * 100:.0f} percent of commercial")
                    if industrial_ratio > 0 and round(industrial_ratio * 100) > 0:
                        landuse_descriptions.append(f"{industrial_ratio * 100:.0f} percent of industrial")
                    if parking_ratio > 0 and round(parking_ratio * 100) > 0:
                        landuse_descriptions.append(f"{parking_ratio * 100:.0f} percent of open parking")
                    if recreational_ratio > 0 and round(recreational_ratio * 100) > 0:
                        landuse_descriptions.append(f"{recreational_ratio * 100:.0f} percent of park")
                    if natural_ratio > 0 and round(natural_ratio * 100) > 0:
                        landuse_descriptions.append(f"{natural_ratio * 100:.0f} percent of nature reserve")

                    # 拼接 land use 描述（如有）
                    landuse_text = ""
                    if landuse_descriptions:
                        landuse_text = "Land use parcels include: " + ", ".join(landuse_descriptions) + "."

                    # 拼接最终 prompt
                    prompt = (
                        f"Satellite imagery of {city}. "
                        f"The Building Coverage Ratio in this area is {Building_Coverage_Ratio:.2f}%. "
                        f"The Building Volume Density is {Building_Volume_Density:.2f} cubic meters per square meter. "
                        f"The Road Density is {road_density} kilometers per square kilometer."
                        f"{landuse_text}"
                    )

                results.append({'prompt': prompt, 'target': os.path.join(directory, filename),
                                'source': os.path.join(directory_hint, filename_hint),
                                'dem': os.path.join(directory_dem, filename_dem)})


                if n1<road_density:
                    n1 = road_density
                # if n2<Residential_Volume_Per_Capita:
                #     n2 = Residential_Volume_Per_Capita
                if n3<Building_Coverage_Ratio:
                    n3 = Building_Coverage_Ratio
                if n4<Building_Volume_Density:
                    n4 = Building_Volume_Density
            else:
                print(f"{filename_hint}: 不存在")
        else:
            print(f"{filename}: 不存在")
    # print("RVP>1000:",RVP_count)
    if count_complete+count_incomplete>0:
        print('missing data',count_incomplete/(count_complete+count_incomplete))
    else:
        print('no data in this area')
    return results

# 检查每个GeoJSON文件
all_results = []

cities_number = 0
for city in cities:
    city_results=[]
    city_path = os.path.join(base_path, city)
    print(city_path)
    geojson_files = [os.path.join(city_path, f) for f in os.listdir(city_path) if f.endswith('.geojson') and 'water' in f]
    print(geojson_files)
    cities_number=cities_number+1
    # if cities_number>10:
    #     continue

    total_count = 0
    for geojson_path in geojson_files:
        # 从文件名提取r和d的值，构建正确的图片目录名
        filename_parts = geojson_path.split('/')[-1].replace('.geojson', '').split('_')

        # r_value = filename_parts[3]  # 提取r的值
        # d_value = filename_parts[4]  # 提取d的值
        # directory_name = f"{r_value}_{d_value}_cv"
        directory_path = os.path.join(city_path, 'satellite_image')
        # directory_name_hint = f"{r_value}_{d_value}"
        directory_path_hint = os.path.join(city_path, 'hint_image')
        # directory_name_dem = f"{r_value}_{d_value}"
        directory_path_dem = os.path.join(city_path, 'dem_image_jpg')


        landuse_path = geojson_path.split('/')[-1].split('_')  # 提取文件名（不含路径）
        landuse_path = '_'.join(landuse_path[:3] + ['density', 'landuse.geojson'])  # 拼接成新的文件名
        landuse_path = os.path.join(city_path, landuse_path)

        if os.path.exists(landuse_path):
            print(f"文件存在: {landuse_path}")
        else:
            print(f"文件不存在: {landuse_path}")
        # 确保目录存在
        if not os.path.exists(directory_path):
            print(f"目录 {directory_path} 不存在")
            continue

        if not os.path.exists(directory_path_hint):
            print(f"目录 {directory_path_hint} 不存在")
            continue

        if not os.path.exists(directory_path_dem):
            print(f"目录 {directory_path_dem} 不存在")
            continue

        # 加载GeoJSON数据
        with open(landuse_path, 'r') as file:
            geojson_data = json.load(file)

        # 生成图片文件名
        image_filenames = [
            (f"{feature['properties']['row']}_{feature['properties']['col']}_{feature['properties']['r']}_{feature['properties']['d']}.jpg",
             f"{feature['properties']['row']}_{feature['properties']['col']}_{feature['properties']['r']}_{feature['properties']['d']}.tif",
             f"{feature['properties']['row']}_{feature['properties']['col']}_{feature['properties']['r']}_{feature['properties']['d']}.jpg",feature)
            for feature in geojson_data['features']
        ]



        result=check_files_existence(directory_path, directory_path_hint,directory_path_dem,image_filenames, city)

        city_results.extend(result)

        # Creating a sample of random 2000 elements from the original list of 2254
        total_count = len(city_results)

    np.random.seed(0)  # For reproducibility
    if len(city_results) > 5000:
        city_results = np.random.choice(city_results, 2000, replace=True)
    all_results.extend(city_results)

    print(f"Total files found for {city}: {total_count}")


    # with open('./urban_data/tencities/all.json', 'w') as outfile:
    #     for result in all_results:
    #         outfile.write(json.dumps(result) + '\n')


    random.shuffle(all_results)

    # 计算划分点
    split_point = int(len(all_results) * 0.8)

    # 划分数据
    train_data = all_results[:split_point]
    test_data = all_results[split_point:]

    # Save results to a JSON file
    with open('./train.json', 'w') as outfile:
        for result in train_data:
            outfile.write(json.dumps(result) + '\n')

    with open('./test.json', 'w') as outfile:
        for result in test_data:
            outfile.write(json.dumps(result) + '\n')

print(n1,n2,n3,n4)