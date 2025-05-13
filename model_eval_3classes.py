import torch
import torch.nn.functional as F

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

# from energy_loss_2class import semantic_energy_loss

import h5py
import cv2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from  skimage.feature import peak_local_max
import numpy as np

import numpy as np
import scipy.spatial as S
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def binary_match(pred_points, gd_points, threshold_distance=18):
    dis = S.distance_matrix(pred_points, gd_points)
    connection = np.zeros_like(dis)
    connection[dis < threshold_distance] = 1
    graph = csr_matrix(connection)
    res = maximum_bipartite_matching(graph, perm_type='column')
    right_points_index = np.where(res > 0)[0]
    right_num = len(right_points_index)

    matched_gt_points = res[right_points_index]

    if len(np.unique(matched_gt_points)) != len(matched_gt_points):
        import pdb;
        pdb.set_trace()

    return right_num, right_points_index


# model_state_dict=torch.load('/home/xuzhengyang/code/vig_model_test_2023-04-21_23_57_31.pkl')
# model_state_dict=torch.load('/home/xuzhengyang/code/vig_model_test_2023-04-23_15_12_47.pkl')
# model_state_dict=torch.load('/home/xuzhengyang/code/vig_model_test_2023-04-23_16_21_35.pkl')
# model_state_dict=torch.load('/home/xuzhengyang/code/vig_model_test_2023-04-23_17_27_14.pkl')
# model_state_dict=torch.load('/home/xuzhengyang/code/vig_model_test_epoch_2023-05-04_22_19_04.pkl')
model_state_dict=torch.load('/home/xuzhengyang/code/vig_multi_class/vig_model_tumor_epoch_50_time_2023-12-29_21_02_26.pkl')
threshold=0.5


vig_model=vig_s_224_gelu()
# vig_model=vig_ti_224_gelu()

vig_model.load_state_dict(model_state_dict)
vig_model.cuda()
vig_model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# train_dataset = Cell_Dataset(data_root='/home/data/xuzhengyang/ki67/img/train', gt_root='/home/data/xuzhengyang/ki67/ground_truth/train', transform=transform)
# train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# for images, class_heat_maps, position_heat_maps, nodes_label in train_dataloader:
#         images=images.cuda()
#         output_feature,regression,classification = vig_model.forward(images)
#         print(images)
# img_path='/home/data/xuzhengyang/ki67/img/test/42.png'
# h5_path='/home/data/xuzhengyang/ki67/ground_truth_2classes/test/42.h5'
img_path='/home/data/xuzhengyang/Her2/img/train/21.png'
h5_path='/home/data/xuzhengyang/Her2/ground_truth_3classes_new/train/21.h5'
images = Image.open(img_path)
images = transform(images)
images=images.cuda()
C,H,W=images.shape
images=images.reshape(1,C,H,W)

with h5py.File(h5_path, 'r') as hf:
    class_heat_map = np.array(hf.get('class_heat_map'))
    nodes_label=np.array(hf.get('sparse_target'))
    sparse_class_heat_map=np.array(hf.get('sparse_class_heat_map'))
label_coordinates=peak_local_max(class_heat_map[1], min_distance=10,  exclude_border=6 // 2)
label_coordinates_tissue=peak_local_max(class_heat_map[2], min_distance=10,  exclude_border=6 // 2)
sparse_label_coordinates=peak_local_max(sparse_class_heat_map[1], min_distance=6,  exclude_border=6 // 2)
# print(type(label_coordinates))
print(label_coordinates.shape)


sparse_node_label=np.zeros(nodes_label.shape)
full_node_label=np.zeros(nodes_label.shape)

for i in range(sparse_label_coordinates.shape[0]):
    x=int(sparse_label_coordinates[i][0]/32)
    y=int(sparse_label_coordinates[i][1]/32)
    sparse_node_label[x][y]=9

for i in range(label_coordinates.shape[0]):
    x=int(label_coordinates[i][0]/32)
    y=int(label_coordinates[i][1]/32)
    full_node_label[x][y]=9

with torch.no_grad():
    # output_feature,regression,_,_ = vig_model.forward(images)
    output_feature,regression= vig_model.forward(images)

regression=regression.reshape(3,H,W)

regression=regression.cpu().detach().numpy()

store_name='/home/xuzhengyang/code/vig_multi_class/result/21_epoch_2023-12-29_21_02_26.h5'
with h5py.File(store_name, 'w') as f:
        # 将jgPesult写入HDF5文件
        f.create_dataset('regression', data=regression)



cell_mask=regression[1]
cell_mask[cell_mask<threshold]=0
tissue_mask=regression[2]
tissue_mask[tissue_mask<threshold]=0

min_len=6
coordinates=peak_local_max(cell_mask, min_distance=min_len,  exclude_border=min_len // 2)
coordinates_tissue=peak_local_max(tissue_mask, min_distance=min_len,  exclude_border=min_len // 2)

print(coordinates.shape)

# right_num,_=binary_match(coordinates,label_coordinates)
# print(right_num)

# image = cv2.imread(img_path)



# # # 遍历坐标点列表，绘制每个点
# for point in label_coordinates:
#     x, y = point
#     # 绘制圆形点（半径为2）
#     cv2.circle(image, (y, x), 5, (255, 0, 0), -1)#蓝色是所有肿瘤细胞标注点类别

# for point in label_coordinates_tissue:
#     x, y = point
#     # 绘制圆形点（半径为2）
#     cv2.circle(image, (y, x), 5, (0, 255, 0), -1)#绿色是所有组织细胞标注点类别


# # for point in sparse_label_coordinates:
# #     x, y = point
# #     # 绘制圆形点（半径为2）
# #     cv2.circle(image, (y, x), 5, (0, 255, 0), -1)#绿色是稀疏标注点


# for point in coordinates:
#     x, y = point
#     # 绘制圆形点（半径为2）
#     cv2.circle(image, (y, x), 4, (0, 0, 255), -1)#红色是预测点

# for point in coordinates_tissue:
#     x, y = point
#     # 绘制圆形点（半径为2）
#     cv2.circle(image, (y, x), 4, (0, 0, 255), -1)#红色是预测点


# # 显示绘制了坐标点的图像
# # cv2.imshow("Image with Points", image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# cv2.imwrite("/home/xuzhengyang/code/multi-class-sparse-detection/result/21_image_with_points_2023-10-07_15_14_23_0.5.jpg", image)

