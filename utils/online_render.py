import os
import numpy as np
from utils.recon_helpers import setup_camera
from PIL import Image, ImageDraw, ImageFont
import time
import torch
from diff_gaussian_rasterization import GaussianRasterizer as Renderer


# global vis utils, ugly code used only for personal debugging
w2cs = np.load('w2cs.npy', allow_pickle=True)
w2ci = 0
total_cnt = 0
fix_v = None

def online_render(curr_data, iter_time_idx, rendervar, dev_use_controller=False):
    global w2ci, total_cnt, fix_v
    if os.path.exists('./online_vis'):
        if not dev_use_controller and os.path.exists('./online_vis/'+str(iter_time_idx).zfill(6)+'.png'):
            return
        
        vis_cam = curr_data['cam']
        
        if not dev_use_controller:
            try: 
                fix_v = w2cs[w2ci]
            except IndexError:
                fix_v = w2cs[-1]
            except Exception as e:
                pass
        else:
            try:
                fix_v = np.load('pose.npy', allow_pickle=True)
            except Exception as e:
                print(e)
        w2ci += 1
        vis_cam = setup_camera(vis_cam.image_width, vis_cam.image_height, curr_data['intrinsics'].cpu().numpy(), fix_v, bg=[1, 1, 1])
        st = time.perf_counter()
        im, _, _ = Renderer(raster_settings=vis_cam)(**rendervar)
        im = torch.clamp(im, 0, 1)
        rt = time.perf_counter() - st
        rt_ms = f"Render time: {rt * 1000:.2f} ms"
        rimg = Image.fromarray(np.uint8((torch.permute(im, (1, 2, 0))).detach().cpu().numpy()*255), 'RGB')
        draw = ImageDraw.Draw(rimg)
        font = ImageFont.truetype("arial.ttf", 40)
        # draw.text((10, 10), rt_ms, fill=(255, 255, 255), font=font)
        if not dev_use_controller:
            rimg.save('./online_vis/'+str(iter_time_idx).zfill(6)+'.png') # should be 0000xx.png
        else:
            rimg.save('./online_vis/'+str(total_cnt).zfill(6)+'.png')
            total_cnt += 1
        