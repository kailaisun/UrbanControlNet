import json
import os
import random
# Munich文件夹的路径，包含GeoJSON文件和图片目录
# GenAI_density文件夹的路径
base_path = './urban_data/tencities/GenAI_density/'

# 获取所有城市的目录
cities = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

n1=0
n2=0
n3=0
n4=0
def check_files_existence(directory,directory_hint, filenames,city):
    global n1,n2,n3,n4
    existing_files = os.listdir(directory)
    existing_files_hint=os.listdir(directory_hint)
    count=0
    RVP_count=0
    results = []
    for filename, filename_hint,feature in filenames:
        if filename in existing_files:
            if filename_hint in existing_files_hint:
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


                Building_Coverage_Ratio=round(built_surface_total/land*100,2)
                Building_Volume_Density=round(built_volume_total/land,2)


                if Residential_Volume_Per_Capita>1000:
                    RVP_count=RVP_count + 1
                    Residential_Volume_Per_Capita=1000
                prompt = (f"Satellite imagery of {city}. The Residential Population Density in this area is {Residential_Population_Density} persons per thousand square meter. "
                          f"The Residential Volume Per Capita in this area is {Residential_Volume_Per_Capita} cubic meters per person. "
                          f"The Building Coverage Ratio in this area is {Building_Coverage_Ratio} %."
                          f"The Building Volume Density in this area is {Building_Volume_Density} cubic meters per square meter.")

                # prompt = (f"Satellite imagery of {city}. The total built-up surface area is {built_surface_total} thousand square meters. "
                #           f"The total built-up volume is {built_volume_total} thousand cubic meters. "
                #           f"The population in this area is about {population_count}.")
                results.append({'prompt': prompt, 'target': os.path.join(directory, filename),'source': os.path.join(directory_hint, filename_hint)})
                if n1<Residential_Population_Density:
                    n1 = Residential_Population_Density
                if n2<Residential_Volume_Per_Capita:
                    n2 = Residential_Volume_Per_Capita
                if n3<Building_Coverage_Ratio:
                    n3 = Building_Coverage_Ratio
                if n4<Building_Volume_Density:
                    n4 = Building_Volume_Density
            else:
                print(f"{filename_hint}: 不存在")
        else:
            print(f"{filename}: 不存在")
    print("RVP>1000:",RVP_count)
    return results

# 检查每个GeoJSON文件
all_results = []
for city in cities:
    city_path = os.path.join(base_path, city)
    print(city_path)
    geojson_files = [os.path.join(city_path, f) for f in os.listdir(city_path) if f.endswith('.geojson') and 'water' in f]
    print(geojson_files)

    total_count = 0
    for geojson_path in geojson_files:
        # 从文件名提取r和d的值，构建正确的图片目录名
        filename_parts = geojson_path.split('/')[-1].replace('.geojson', '').split('_')

        r_value = filename_parts[3]  # 提取r的值
        d_value = filename_parts[4]  # 提取d的值
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


    # with open('./urban_data/tencities/all.json', 'w') as outfile:
    #     for result in all_results:
    #         outfile.write(json.dumps(result) + '\n')


    random.shuffle(all_results)

    # 计算划分点
    split_point = int(len(all_results) * 0.7)

    # 划分数据
    train_data = all_results[:split_point]
    test_data = all_results[split_point:]

    # Save results to a JSON file
    # with open('./urban_data/tencities/train.json', 'w') as outfile:
    #     for result in train_data:
    #         outfile.write(json.dumps(result) + '\n')
    #
    # with open('./urban_data/tencities/test.json', 'w') as outfile:
    #     for result in test_data:
    #         outfile.write(json.dumps(result) + '\n')

print(n1,n2,n3,n4)