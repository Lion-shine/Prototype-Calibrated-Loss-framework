from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import glob
import h5py

from  skimage.feature import peak_local_max

def gaussian(x, mean, sigma):
    return np.exp(-np.power(x - mean, 2.) / (2 * np.power(sigma, 2.)))

def generate_heatmap(x, y, radius, width, height):
    x = int(x)
    y = int(y)
    y_indices, x_indices = np.indices((height, width))
    dist = np.sqrt((y_indices - y) ** 2 + (x_indices - x) ** 2)
    mask = dist <= radius
    heatmap = np.zeros((height, width))
    heatmap[mask] = gaussian(dist[mask], 0, radius / 2)
    return heatmap

def generate_gaussian_heatmap(points, radius, width, height):
    y_indices, x_indices = np.indices((height, width))
    heatmap = np.zeros((height, width))
    for point in points:
        x, y = point
        # x=x/8
        # y=y/8
        x=x/16
        y=y/16
        heatmap += generate_heatmap(y, x, radius, width, height)  #可视化要修改坐标位置
    return heatmap

class Cell_Dataset(Dataset):
    def __init__(self, data_root, gt_root, transform=None):
        self.root1 = data_root
        self.root2 = gt_root
        self.transform = transform
        # self.image_files = sorted(os.listdir(data_root))
        # 只加载图像文件（支持 .jpg, .png, .jpeg）
        self.image_files = sorted([f for f in os.listdir(data_root) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.h5_files = sorted(os.listdir(gt_root))


    def __getitem__(self, index):
        # load image
        img_path = os.path.join(self.root1, self.image_files[index])
        img = Image.open(img_path)
        # print(img_path)


        # load ground truth data
        h5_path = os.path.join(self.root2, self.h5_files[index])

        with h5py.File(h5_path, 'r') as hf:
            class_heat_map = np.array(hf.get('sparse_class_heat_map'))
            full_calss_heat_map=np.array(hf.get('class_heat_map'))
            nodes_label=np.array(hf.get('sparse_target'))

        # sparse_label_coordinates1=peak_local_max(class_heat_map[1], min_distance=6,  exclude_border=6 // 2)   #肿瘤细胞
        # sparse_label_coordinates2=peak_local_max(class_heat_map[2], min_distance=6,  exclude_border=6 // 2)   #组织细胞
        
        # # nodes_label=np.zeros((2,64,64))
        # # nodes_label[0] = generate_gaussian_heatmap(sparse_label_coordinates1, radius=2, width=64, height=64)
        # # nodes_label[1] = generate_gaussian_heatmap(sparse_label_coordinates2, radius=2, width=64, height=64)
        # # nodes_label=np.clip(nodes_label,0.0,1.0)
        # # print(sparse_label_coordinates)
        # nodes_label=np.zeros(nodes_label_formal.shape)

        # for i in range(sparse_label_coordinates1.shape[0]):
        #     x=int(sparse_label_coordinates1[i][0]/32)
        #     y=int(sparse_label_coordinates1[i][1]/32)
        #     nodes_label[x][y]=1

        # for i in range(sparse_label_coordinates2.shape[0]):
        #     x=int(sparse_label_coordinates2[i][0]/32)
        #     y=int(sparse_label_coordinates2[i][1]/32)
        #     nodes_label[x][y]=2

        # nodes_label64=np.zeros((64,64))

        # for i in range(sparse_label_coordinates1.shape[0]):
        #     x=int(sparse_label_coordinates1[i][0]/16)
        #     y=int(sparse_label_coordinates1[i][1]/16)
        #     nodes_label64[x][y]=1

        # for i in range(sparse_label_coordinates2.shape[0]):
        #     x=int(sparse_label_coordinates2[i][0]/16)
        #     y=int(sparse_label_coordinates2[i][1]/16)
        #     nodes_label64[x][y]=2

        # nodes_label128=np.zeros((128,128))

        # for i in range(sparse_label_coordinates1.shape[0]):
        #     x=int(sparse_label_coordinates1[i][0]/8)
        #     y=int(sparse_label_coordinates1[i][1]/8)
        #     nodes_label128[x][y]=1

        # for i in range(sparse_label_coordinates2.shape[0]):
        #     x=int(sparse_label_coordinates2[i][0]/8)
        #     y=int(sparse_label_coordinates2[i][1]/8)
        #     nodes_label128[x][y]=2

        if self.transform is not None:
            img = self.transform(img)
        # print(img.shape)
            

        return img, class_heat_map, full_calss_heat_map, nodes_label#,nodes_label64,nodes_label128

    def __len__(self):
        return len(self.image_files)


# transform = transforms.Compose([
#     # transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# dataset = Cell_Dataset(data_root='/home/xuzhengyang/code/vig_pytorch/data/img', gt_root='/home/xuzhengyang/code/vig_pytorch/data/ground_truth', transform=transform)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# for images, class_heat_maps, position_heat_maps in dataloader:
#     print(images.shape)
#     print(class_heat_maps.shape)
#     print(position_heat_maps.shape)
'''
    torch.Size([10, 10, 1024, 1024])   images
    torch.Size([10, 2, 1024, 1024])    class_heat_maps
    torch.Size([10, 10, 1024, 1024])   position_heat_maps
    '''