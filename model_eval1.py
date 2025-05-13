import torch

# from energy_loss1 import semantic_energy_loss
# from vig2 import vig_ti_224_gelu,vig_s_224_gelu
from vig1 import vig_ti_224_gelu,vig_s_224_gelu

from torch.utils.data import DataLoader
from celldataset import Cell_Dataset

import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

from energy_loss8 import semantic_energy_loss

import h5py
matplotlib.use('agg')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# model_state_dict=torch.load('/home/xuzhengyang/code/vig_model_test_2023-04-21_23_57_31.pkl')
# model_state_dict=torch.load('/home/xuzhengyang/code/vig_model_test_2023-04-23_15_12_47.pkl')
# model_state_dict=torch.load('/home/xuzhengyang/code/vig_model_test_2023-04-23_16_21_35.pkl')
# model_state_dict=torch.load('/home/xuzhengyang/code/vig_model_test_2023-04-23_17_27_14.pkl')
# model_state_dict=torch.load('/home/xuzhengyang/code/vig_model_test_epoch_2023-05-04_22_19_04.pkl')
model_state_dict=torch.load('/home/xuzhengyang/vig_model_test_epoch_2023-05-10_22_49_07.pkl')



vig_model=vig_s_224_gelu()
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
img_path='/home/data/xuzhengyang/ki67/img/train/21.png'
images = Image.open(img_path)
images = transform(images)
images=images.cuda()
C,H,W=images.shape
images=images.reshape(1,C,H,W)

output_feature,regression,classification = vig_model.forward(images)
regression=regression.reshape(2,H,W)
classification=classification.reshape(4,H,W)
print(regression)
print(classification)

regression=regression.cpu().detach().numpy()
classification=classification.cpu().detach().numpy()


store_name='/home/xuzhengyang/code/vig_pytorch/result/21_epoch_2023-05-10_22_49_07.h5'
with h5py.File(store_name, 'w') as f:
        # 将jgPesult写入HDF5文件
        f.create_dataset('regression', data=regression)

        f.create_dataset('classification', data=classification)
# regression_obj=regression[1].cpu().detach().numpy()
# plt.imshow(regression_obj,cmap='hot',interpolation='nearest')
# plt.colorbar()
# plt.show()

