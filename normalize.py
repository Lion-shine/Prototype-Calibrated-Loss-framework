
# from energy_loss1 import semantic_energy_loss
# from vig2 import vig_ti_224_gelu,vig_s_224_gelu
from vig_3classes3 import vig_ti_224_gelu,vig_s_224_gelu
# from vig_2classes3 import vig_ti_224_gelu,vig_s_224_gelu

from torch.utils.data import DataLoader
from celldataset import Cell_Dataset

import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

import numpy as np

# from energy_loss_2class import semantic_energy_loss

import h5py
import cv2

import os

h5_path='/home/xuzhengyang/code/vig_multi_class/vis/heatmap2.h5'
with h5py.File(h5_path, 'r') as hf:
    regression=np.array(hf.get('regression'))


normailze=(regression-np.min(regression))/(np.max(regression)-np.min(regression))
print(normailze)

store_name='/home/xuzhengyang/code/vig_multi_class/vis/normailze2.h5'
with h5py.File(store_name, 'w') as f:
        # 将jgPesult写入HDF5文件
        f.create_dataset('regression', data=normailze)