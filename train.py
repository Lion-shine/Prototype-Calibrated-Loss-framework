import torch
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler

import argparse

from energy_loss_3classes import semantic_energy_loss

from vig_3classes import vig_ti_224_gelu, vig_s_224_gelu

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

    parser.add_argument("--img_dir", type=str, default='/data/img_tumor/train', help="your training image path")
    parser.add_argument("--gt_dir", type=str, default='/data/groud_truth_tumor_50/train', help="your training image path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=300)

    return parser

def main():
    parser = get_arguments()
    print(f"[INFO] Parsing command-line arguments: {parser}")

    args = parser.parse_args()
    print(f"[INFO] Parsed arguments: {args}")
    
    # Check if dataset paths exist
    print(f"[INFO] Checking image directory: {args.img_dir}")
    print(f"[INFO] Checking label directory: {args.gt_dir}")
    
    # Print dataset size
    train_dataset = Cell_Dataset(data_root=args.img_dir, gt_root=args.gt_dir, transform=transform)
    print(f"[INFO] Training dataset size: {len(train_dataset)}")
    
    # Print shape of the first sample (for debugging)
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"[INFO] Sample shapes - Image: {sample[0].shape}, Sparse heatmap: {sample[1].shape}, Full heatmap: {sample[2].shape}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"[INFO] Number of batches in training dataloader: {len(train_dataloader)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    if device.type == 'cuda':
        print(f"[INFO] GPU name: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    vig_model = vig_s_224_gelu()
    vig_model.cuda()
    print(f"[INFO] Number of model parameters: {sum(p.numel() for p in vig_model.parameters())}")

    epochs = args.epoch
    vig_model.train()
    optimizer = torch.optim.Adam(vig_model.parameters(), lr=0.01)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150], gamma=0.1)
    se_loss = semantic_energy_loss()

    print(f"[INFO] Starting training, total epochs: {epochs}")
    
    for epoch in tqdm(range(epochs), desc='Epoch'):
        epoch_loss = 0
        mean_regression = 0
        mean_stretched_feature = 0
        mean_exclusive = 0
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[INFO] Epoch {epoch}, learning rate: {current_lr}")
        
        # Monitor data loading time
        data_load_start = datetime.now()
        for i, (images, sparse_class_heat_maps, full_calss_heat_map, nodes_label) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch} - Batch')):
            if i == 0:  # Print info for first batch only to avoid clutter
                print(f"[INFO] Batch {i} - Image shape: {images.shape}, Sparse heatmap shape: {sparse_class_heat_maps.shape}")
            
            # Measure data loading time
            if i == 0 and epoch == 0:
                data_load_time = (datetime.now() - data_load_start).total_seconds()
                print(f"[INFO] Time to load first batch: {data_load_time:.2f} seconds")
            
            optimizer.zero_grad()
            images = images.cuda()
            
            sparse_class_heat_maps = torch.clip(sparse_class_heat_maps, 0.0, 1.0)
            class_heat_maps = sparse_class_heat_maps.cuda()

            # Monitor forward pass time
            forward_start = datetime.now()
            output_feature, regression = vig_model.forward(images)
            forward_time = (datetime.now() - forward_start).total_seconds()
            
            # Print output shapes and forward time
            if i == 0:
                print(f"[INFO] Model output - Feature shape: {output_feature.shape}, Regression shape: {regression.shape}")
                print(f"[INFO] Forward pass time: {forward_time:.2f} seconds")

            # Monitor loss calculation time
            loss_start = datetime.now()
            loss, regression_loss, stretched_feature_loss, exclusion_loss = se_loss.forward(output_feature, nodes_label, regression, class_heat_maps)
            loss_time = (datetime.now() - loss_start).total_seconds()
            
            # Print loss values and time
            if i % 10 == 0:
                print(f"[INFO] Epoch {epoch}, Batch {i} - Loss: {loss.item():.4f}, Regression loss: {regression_loss.item():.4f}, Feature loss: {stretched_feature_loss.item():.4f}, Exclusion loss: {exclusion_loss.item():.4f}")
                print(f"[INFO] Loss calculation time: {loss_time:.2f} seconds")

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss
            mean_regression += regression_loss
            mean_stretched_feature += stretched_feature_loss
            mean_exclusive += exclusion_loss
            
            # Check GPU memory usage
            if i % 50 == 0 and device.type == 'cuda':
                print(f"[INFO] GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        epoch_loss /= len(train_dataloader)
        mean_regression /= len(train_dataloader)
        mean_stretched_feature /= len(train_dataloader)
        mean_exclusive /= len(train_dataloader)

        tqdm.write(f"Epoch {epoch}: Loss={epoch_loss:.4f}, regression_Loss={mean_regression:.4f}, stretched_feature_loss={mean_stretched_feature:.4f}, exclusive={mean_exclusive:.4f}")

        if epoch % 50 == 0 and epoch != 0:
            now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            save_name = f'results/vig_her2_model_tumor_epoch_{epoch}_time_{now}.pkl'
            torch.save(vig_model.state_dict(), save_name)
            print(f"[INFO] Model saved: {save_name}")

    now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    save_name = f'results/vig_her2_model_tumor_final_{epoch}_time_{now}.pkl'
    torch.save(vig_model.state_dict(), save_name)
    print(f"[INFO] Final model saved: {save_name}")
    print(f"[INFO] Training complete, total epochs: {epochs}")

if __name__ == "__main__":
    main()
