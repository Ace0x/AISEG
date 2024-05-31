from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmseg.utils import get_palette
import mmcv
import matplotlib as plt
import numpy as np

config_file = '../work_dirs/deeplabv3_r50-d8_4xb2-40k_LandClass-512x1024/deeplabv3_r50-d8_4xb2-40k_LandClass-512x1024.py'
checkpoint_file = '../work_dirs/deeplabv3_r50-d8_4xb2-40k_LandClass-512x1024/iter_10000.pth'

img = mmcv.imread('../data/land_cover/img_dir/val/878990_sat.jpg')
palette=[[0,0,0],[0,255,255],[255,255,0],[255,0,255],[0,255,0],[0,0,255],[255,255,255]]
model = init_model(config_file, checkpoint_file, device='cuda:0')
result = inference_model(model, img)
pred = np.asarray(result)

show_result_pyplot(model,img,result)