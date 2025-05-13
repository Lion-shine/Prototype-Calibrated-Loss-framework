import torch
import torch.nn.functional as F


from vig_3classes import vig_ti_224_gelu,vig_s_224_gelu

from torch.utils.data import DataLoader
from celldataset import Cell_Dataset

import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib


import h5py
import cv2

import os

from  skimage.feature import peak_local_max
import numpy as np

import numpy as np
import scipy.spatial as S
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def binary_match(pred_points, gd_points, threshold_distance=15):
    dis = S.distance_matrix(pred_points, gd_points)
    connection = np.zeros_like(dis)
    connection[dis < threshold_distance] = 1
    graph = csr_matrix(connection)
    res = maximum_bipartite_matching(graph, perm_type='column')
    right_points_index = np.where(res > 0)[0]
    right_num = right_points_index.shape[0]

    matched_gt_points = res[right_points_index]
    text=np.unique(matched_gt_points)



    if (np.unique(matched_gt_points)).shape[0] != (matched_gt_points).shape[0]:
        import pdb;
        pdb.set_trace()

    return right_num, right_points_index


threshold=0.1
vig_pdl1_model_tumor_epoch_350_time_2024-09-10_21_57_36.pkl') 

# 定义图像和掩码的根目录10,30
img_root = 'autodl-tmp/ocelot2023_v1.0.1/images/test/cell'
mask_root = 'autodl-tmp/ocelot2023_v1.0.1/mask/test'
model_state_dict=torch.load('autodl-tmp/saved_models_70/vig_model_tumor_epoch_250_loss_0.0776_time_2025-03-12_08_26_44.pkl') 

vig_model=vig_s_224_gelu()
vig_model.load_state_dict(model_state_dict)
vig_model.cuda()
vig_model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

len=0
total_tumor_precision=0
total_tumor_recall=0
total_other_precision=0
total_other_recall=0
tumor_cnt=0
other_cnt=0


# 修改遍历逻辑
for img_file in os.listdir(img_root):
    img_path = os.path.join(img_root, img_file)
    if not os.path.isfile(img_path) or img_file.startswith('.'):
        print(f"Skipping directory or hidden file: {img_path}")
        continue
    
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        print(f"Processing file: {img_path}")
        len = len + 1
    
        
        h5_filename = os.path.splitext(img_file)[0] + '.h5'  
        h5_path = os.path.join(mask_root, h5_filename)
    
        
        
        images = Image.open(img_path)
        images = transform(images)
        images=images.cuda()
        C,H,W=images.shape
        images=images.reshape(1,C,H,W)
    
        with h5py.File(h5_path, 'r') as hf:
            class_heat_map = np.array(hf.get('class_heat_map'))
            nodes_label=np.array(hf.get('sparse_target'))
            sparse_class_heat_map=np.array(hf.get('sparse_class_heat_map'))
        tumor_label_coordinates=peak_local_max(class_heat_map[1], min_distance=12,  exclude_border=6 // 2)
        other_label_coordinates=peak_local_max(class_heat_map[2], min_distance=12,  exclude_border=6 // 2)
    
    
    
        print('================================')
        print((tumor_label_coordinates).shape[0])
        print((other_label_coordinates).shape[0])
    
        tumor_num=(tumor_label_coordinates).shape[0]
        other_num=(other_label_coordinates).shape[0]
            
        with torch.no_grad():
            output_feature,regression = vig_model.forward(images)
        regression=regression.reshape(3,H,W)
    
        regression=regression.cpu().detach().numpy()
    
        #caculate precision and recall for tumor cell
        tumor_mask=regression[1]
        tumor_mask[tumor_mask<threshold]=0
        min_len=12
        tumor_coordinates=peak_local_max(tumor_mask, min_distance=min_len,  exclude_border=min_len // 2)
    
        tumor_right_num,_=binary_match(tumor_coordinates,tumor_label_coordinates)
        tumor_precision = 0
        tumot_recall = 0
        if tumor_num>=10:
            tumor_precision=tumor_right_num/(tumor_coordinates.shape[0]+1e-10)
            total_tumor_precision=total_tumor_precision+tumor_precision
            
            tumot_recall=tumor_right_num/(tumor_label_coordinates.shape[0]+1e-10)
            total_tumor_recall=total_tumor_recall+tumot_recall
            
            tumor_cnt=tumor_cnt+1    
        print("tumor precision is "+str(tumor_precision))
        print("tumor recall is "+str(tumot_recall))
        
        
    
    
        # caculate precision and recall for tissue cell

        other_mask=regression[2]
        other_mask[other_mask<threshold]=0
        min_len=12
        other_coordinates=peak_local_max(other_mask, min_distance=min_len,  exclude_border=min_len // 2)
    
        other_right_num,_=binary_match(other_coordinates,other_label_coordinates)
    
        if other_num>=10:
            other_precision=other_right_num/(other_coordinates.shape[0]+1e-10)
            other_recall=other_right_num/(other_label_coordinates.shape[0]+1e-10)
        
            total_other_precision=total_other_precision+other_precision
            total_other_recall=total_other_recall+other_recall
            other_cnt=other_cnt+1
    
        print("other precision is "+str(other_precision))
        print("other recall is "+str(other_recall))


total_tumor_precision=total_tumor_precision/tumor_cnt
total_tumor_recall=total_tumor_recall/tumor_cnt
F1_tumor_score=(2*total_tumor_precision*total_tumor_recall)/(total_tumor_precision+total_tumor_recall+1e-10)

total_other_precision=total_other_precision/other_cnt
total_other_recall=total_other_recall/other_cnt
F1_other_score=(2*total_other_precision*total_other_recall)/(total_other_precision+total_other_recall+1e-10)

print("====================")
print("total tumor precision is "+str(total_tumor_precision))
print("total tumor recall is "+str(total_tumor_recall))
print("F1 tumor score is "+str(F1_tumor_score))

print("total other precision is "+str(total_other_precision))
print("total other recall is "+str(total_other_recall))
print("F1 other score is "+str(F1_other_score))

F1_mean_score=(F1_tumor_score+F1_other_score)/2

print("F1 mean score is "+str(F1_mean_score))