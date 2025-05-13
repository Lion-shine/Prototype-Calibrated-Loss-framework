import torch
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler

import argparse

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

def get_arguments():
    parser = argparse.ArgumentParser(description="semantic_energy")

    parser.add_argument("--img_dir", type=str, default='/home/data/xuzhengyang/Her2/img_2classes/train', help="your training image path")
    parser.add_argument("--gt_dir", type=str, default='/home/data/xuzhengyang/Her2/ground_truth_2classes_new/train')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=500)

    return parser

def main():
    parser = get_arguments()
    print(parser)

    args = parser.parse_args()

    train_dataset = Cell_Dataset(data_root=args.img_dir, gt_root=args.gt_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vig_model=vig_s_224_gelu()


    vig_model.cuda()

    epochs = args.epoch
    vig_model.train()
    optimizer = torch.optim.Adam(vig_model.parameters(), lr=0.01)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150], gamma=0.1)


    se_loss=semantic_energy_loss()

    for epoch in tqdm(range(epochs),desc='Epoch'):
        epoch_loss=0
        mean_regression=0
        mean_stretched_feature=0
        mean_exclusive=0

        for images, sparse_class_heat_maps, full_calss_heat_map, nodes_label in tqdm(train_dataloader):
            optimizer.zero_grad()
            images=images.cuda()

            sparse_class_heat_maps=torch.clip(sparse_class_heat_maps,0.0,1.0)
            class_heat_maps=sparse_class_heat_maps.cuda()

        

            output_feature,regression = vig_model.forward(images)

            loss, regression_loss, stretched_feature_loss,exclusion_loss= se_loss.forward(output_feature, nodes_label, regression, class_heat_maps)

            loss.backward()

            optimizer.step()
            scheduler.step()


            epoch_loss=epoch_loss+loss
            mean_regression=mean_regression+regression_loss
            mean_stretched_feature=mean_stretched_feature+stretched_feature_loss
            mean_exclusive=mean_exclusive+exclusion_loss

       
        
        epoch_loss=epoch_loss/len(train_dataloader)
        mean_regression=mean_regression/len(train_dataloader)
        mean_stretched_feature=mean_stretched_feature/len(train_dataloader)
        mean_exclusive=mean_exclusive/len(train_dataloader)

    

    
        tqdm.write(f"Epoch {epoch}: Loss={epoch_loss:.4f},regression_Loss={mean_regression:.4f},stretched_feature_loss={mean_stretched_feature:.4f},exclusive={mean_exclusive:.4f}")


        if epoch%50==0 and epoch != 0:
            now=datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            save_name=f'vig_pdl1_model_tumor_epoch_{epoch}_time_{now}.pkl'
            torch.save(vig_model.state_dict(), save_name)
      

    now=datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    save_name=f'vig_pdl1_model_tumor_final_{epoch}_time_{now}.pkl'
    torch.save(vig_model.state_dict(), save_name)



