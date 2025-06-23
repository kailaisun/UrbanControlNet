from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import pandas as pd

def process(model, x, y, zoom, input_city, output_city, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, v=None, RGB=True):
    ddim_sampler = DDIMSampler(model)
    with torch.no_grad():
        input_image = cv2.imread("../satellite_tiles_control/satellite_tiles/"+str(zoom)+"/"+input_city+"/"+str(zoom)+"_"+str(x)+"_"+str(y)+".png")
        input_image = np.array(input_image)
        H,W,C = input_image.shape

        if v is None:
            detected_map = cv2.imread("../satellite_tiles_control/skeleton/"+str(zoom)+"/"+input_city+"/"+str(x)+"/"+str(y)+".png", cv2.IMREAD_UNCHANGED)
        else:
            detected_map = np.array(detected_map)    

        # convert 4-channel source image to 3-channel
        #make mask of where the transparent bits are
        trans_mask = detected_map[:,:,3] == 0

        #replace areas of transparency with white and not transparent
        detected_map[trans_mask] = [255, 255, 255, 255]

        #new image without alpha channel...
        detected_map = cv2.cvtColor(detected_map, cv2.COLOR_BGRA2BGR)        
        #OpenCV read images in BGR order.
        control = cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)
        control = torch.from_numpy(control.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

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
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        else:
            results = [cv2.cvtColor(x_samples[i], cv2.COLOR_RGB2BGR) for i in range(num_samples)]

        return [detected_map] + [input_image] + results
