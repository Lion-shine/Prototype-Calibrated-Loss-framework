import torch
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler


from energy_loss_3classes import semantic_energy_loss

from vig_3classes import vig_ti_224_gelu,vig_s_224_gelu


from gcn_lib.torch_edge import DenseDilatedKnnGraph
import cv2
import os
import glob
import h5py
from tqdm import tqdm
from datetime import datetime

from torch.utils.data import DataLoader
from celldataset_3classes import Cell_Dataset
import torchvision.transforms as transforms



def acc(full_target, predict):
    matchs = torch.sum(torch.eq(full_target, predict))
    return matchs / full_target.shape[1]


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# train_dataset = Cell_Dataset(data_root='/home/data/xuzhengyang/ki67/img_3classes/train', gt_root='/home/data/xuzhengyang/ki67/ground_truth_3classes_new/train', transform=transform)
# train_dataset = Cell_Dataset(data_root='/home/data/xuzhengyang/Her2/img_tumor/train', gt_root='/home/data/xuzhengyang/Her2/ground_truth_tumor_full/train', transform=transform)
# train_dataset = Cell_Dataset(data_root='/media/ipmi2022/SCSI_all/xuzhengyang/ocelot2023_v1.0.1/images/train/cell', gt_root='/media/ipmi2022/SCSI_all/xuzhengyang/ocelot2023_v1.0.1/mask/train_100', transform=transform)
# train_dataset = Cell_Dataset(data_root='/media/ipmi2022/SCSI_all/xuzhengyang/PDL1_detection/train/images', gt_root='/media/ipmi2022/SCSI_all/xuzhengyang/PDL1_detection/train_full/ground_truth', transform=transform)
train_dataset = Cell_Dataset(data_root='/media/ipmi2022/SCSI_all/xuzhengyang/Her2/img_tumor/train', gt_root='/media/ipmi2022/SCSI_all/xuzhengyang/Her2/ground_truth_tumor_30/train', transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# vig_model = vig_s_224_gelu()#pretrained='/home/xuzhengyang/code/vig_pytorch/vig_ti_74.5.pth')
vig_model=vig_s_224_gelu()
# vig_model=Res18()

# model_state_dict=torch.load('/home/xuzhengyang/code/vig_multi_class/vig_pdl1_model_tumor_epoch_100_time_2024-06-14_15_27_30.pkl')
# vig_model.load_state_dict(model_state_dict)

vig_model.cuda()

epochs = 1000
vig_model.train()
optimizer = torch.optim.Adam(vig_model.parameters(), lr=0.01)#,weight_decay=0.005)
# # # optimizer=torch.optim.SGD(vig_model.parameters(), lr=0.001,weight_decay=0.001)
# # #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 100], gamma=0.1)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150], gamma=0.1)

# optimizer = torch.optim.Adam(vig_model.parameters(), lr=0.0001)#,weight_decay=0.005)


# optimizer = torch.optim.Adam(vig_model.parameters(), lr=0.01)#,weight_decay=0.005)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100,200, 300], gamma=0.1)

se_loss=semantic_energy_loss()

for epoch in tqdm(range(epochs),desc='Epoch'):
    epoch_loss=0
    mean_regression=0
    mean_stretched_feature=0
    mean_exclusive=0

    # mean_l1_loss=0
    for images, sparse_class_heat_maps, full_calss_heat_map, nodes_label in tqdm(train_dataloader):
        optimizer.zero_grad()
        images=images.cuda()
        # sparse_class_heat_maps=torch.clip(full_calss_heat_map,0.0,1.0)

        sparse_class_heat_maps=torch.clip(sparse_class_heat_maps,0.0,1.0)
        class_heat_maps=sparse_class_heat_maps.cuda()

        

        output_feature,regression = vig_model.forward(images)

        # loss, regression_loss, stretched_feature_loss, l1_loss= semantic_energy_loss(output_feature, nodes_label, regression, class_heat_maps)
        loss, regression_loss, stretched_feature_loss,exclusion_loss= se_loss.forward(output_feature, nodes_label, regression, class_heat_maps)

        loss.backward()
        # utils.clip_grad_value_(vig_model.parameters(),clip_value=0.9)x

        optimizer.step()
        scheduler.step()


        epoch_loss=epoch_loss+loss
        mean_regression=mean_regression+regression_loss
        mean_stretched_feature=mean_stretched_feature+stretched_feature_loss
        mean_exclusive=mean_exclusive+exclusion_loss

        # print(loss)

        # mean_l1_loss=mean_l1_loss+l1_loss
        
    epoch_loss=epoch_loss/len(train_dataloader)
    mean_regression=mean_regression/len(train_dataloader)
    mean_stretched_feature=mean_stretched_feature/len(train_dataloader)
    mean_exclusive=mean_exclusive/len(train_dataloader)

    

    # mean_l1_loss=mean_l1_loss/len(train_dataloader)
    
    # print(mean_regression)
    # print(mean_stretched_feature)
    
#     epoch_acc=epoch_acc/len(train_img_paths)
    # tqdm.write(f"Epoch {epoch}: Loss={epoch_loss:.4f},regression_Loss={mean_regression:.4f},stretched_feature_loss={mean_stretched_feature:.4f},l1_loss={mean_l1_loss:.4f}")
    tqdm.write(f"Epoch {epoch}: Loss={epoch_loss:.4f},regression_Loss={mean_regression:.4f},stretched_feature_loss={mean_stretched_feature:.4f},exclusive={mean_exclusive:.4f}")

#     # print("loss is " + str(loss) + "in epoch " + str(epoch))
#     # print("accurate is " + str(accurate))
    if epoch%50==0 and epoch != 0:
        now=datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        save_name=f'vig_pdl1_model_tumor_epoch_{epoch}_time_{now}.pkl'
        torch.save(vig_model.state_dict(), save_name)
      

now=datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
save_name=f'vig_pdl1_model_tumor_final_{epoch}_time_{now}.pkl'
torch.save(vig_model.state_dict(), save_name)



