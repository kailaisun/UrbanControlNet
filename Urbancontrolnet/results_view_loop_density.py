from share import *
import config

import cv2
import einops
#import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from satellite_tiles_density import MyDataset
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# 仅设置一块可见
device = torch.device('cuda:0')
# device=torch.device('cpu')
print(device)  # 输出：device(type='cuda', index=1)


# train_No = 'density_or'
# infer_stage = 4
# infer_num = 200
# desc = 'Description_landuse_med_hh_inc'
test_seed = 42

ckpt_directory = f'./ckpts_s_old_10_4metric/checkpoints_density'
ckpt_epoch_list = ['55'] # '3', '10','30','50','70'
output_file_dir = f'./output_image/ckpts_s_old_10_4metric/epoch_'

def find_epoch_files(directory, epoch_value):
    match_string = f'epoch={epoch_value}-'
    all_files = os.listdir(directory)
    matching_files = [f for f in all_files if f.startswith(match_string)]
    return matching_files

# 查找匹配的文件
ckpt_path_list = []
for d in range(len(ckpt_epoch_list)):
    epoch_value = ckpt_epoch_list[d]
    matching_files = find_epoch_files(ckpt_directory, epoch_value)
    ckpt_path_list.append(f'{ckpt_directory}/{matching_files[0]}')


from satellite_tiles_density import MyDataset
data_dir = './urban_data/tencities/test.json'
dataset = MyDataset(data_dir,paths=True)
print(len(dataset))



def load_model(ckpt_path, device):

    torch.cuda.empty_cache()
    
    # 删除旧模型引用（如果存在）
    if 'model' in globals():
        del globals()['model']
    if 'ddim_sampler' in globals():
        del globals()['ddim_sampler']
    
    # 再次清理 GPU 内存
    torch.cuda.empty_cache()
    # 1. 创建模型实例
    model = create_model('./models/cldm_v15.yaml').cpu()
    
    # 2. 加载模型状态字典
    #ckpt_path = f'./checkpoints/epoch={epoch}-step={step}.ckpt'
    #ckpt_path = f'./version_archive/version_{version}_e{epoch}/checkpoints/epoch={epoch}-step={step}.ckpt'
    model.load_state_dict(load_state_dict(ckpt_path, location='cpu'))
    
    # 3. 将模型移动到指定设备
    model.to(device)
    
    # 4. 创建采样器实例
    ddim_sampler = DDIMSampler(model)
    
    return model, ddim_sampler

def process(hint_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, v=None, RGB=True):
    
    with torch.no_grad():        

        input_image = cv2.imread(hint_path)
        input_image = np.array(input_image)

        print(hint_path)
        print(prompt)
        H,W,C = input_image.shape

        detected_map = cv2.imread(hint_path, cv2.IMREAD_UNCHANGED)
        detected_map = np.array(detected_map)


        
        if detected_map.shape[2] == 4:
            # convert 4-channel source image to 3-channel
            #make mask of where the transparent bits are
            trans_mask = detected_map[:,:,3] == 0
    
            #replace areas of transparency with white and not transparent
            detected_map[trans_mask] = [255, 255, 255, 255]
    
            #new image without alpha channel...
            detected_map = cv2.cvtColor(detected_map, cv2.COLOR_BGRA2BGR)        
        
        #OpenCV read images in BGR order.
        control_1 = cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)
        control_1=cv2.resize(control_1, (512, 512), interpolation=cv2.INTER_AREA)
        control_1 = torch.from_numpy(control_1.copy()).float().to(device) / 255.0

        control=control_1



        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        control=control.to(device)

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
            
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
            
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        if RGB:
            results = [x_samples[i] for i in range(num_samples)]
            detected_map = cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)
        else:
            results = [cv2.cvtColor(x_samples[i], cv2.COLOR_RGB2BGR) for i in range(num_samples)]

        return [detected_map] + [input_image] + results,samples


def convert_bgr_to_rgb(image):
    """
    将 BGR 图像转换为 RGB 图像。
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



num_samples = 3
image_resolution = 512
strength = 1.0
guess_mode = False
detect_resolution = 512
ddim_steps = 20
scale = 9.0
seed = 5354 # [42, 1234, 5678]
eta = 0.0
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'



for e in range(len(ckpt_epoch_list)):
    
    ckpt_path = ckpt_path_list[e]
    model, ddim_sampler = load_model(ckpt_path, device)
    
    outputs = []
    
    output_file = output_file_dir + str(ckpt_epoch_list[e]) #ckpt_list[e]['epoch']
    os.makedirs(output_file, exist_ok=True)

    for i in range(0,len(dataset)): #df.shape[0]
        print(i, )
        image_path,prompt,hint_path = dataset[i]


        
        RGB=True
        # for seed in seeds:
        outputs_i,lfeatures = process(hint_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, v='0', RGB=RGB)
        outputs.append(outputs_i)


        array = lfeatures.cpu().numpy()
        # 保存为NPZ文件
        numpy_file=hint_path[:-3]+"npz"
        np.savez(numpy_file, array=array)
        print(numpy_file)

        ct=np.load(numpy_file)


        for j, image in enumerate(outputs_i[2:]):
            # Save each image with the desired naming convention
            parts = hint_path.split('/')

            # 获取最后三部分并用'-'连接
            new_path = '-'.join(parts[-4:])

            filename = new_path+'_'+str(j)+'.png'
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 将 numpy 数组转换为 PIL 图像
            image_pil = Image.fromarray(image)
            # 保存图像并设置质量和优化参数
            image_pil.save(os.path.join(output_file,filename), format='JPEG', quality=85, optimize=True)
    
