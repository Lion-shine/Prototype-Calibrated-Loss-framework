import h5py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
# from vig_2classes import vig_ti_224_gelu
import numpy as np

#改的是14，把hinge loss给改成了L2正则化，用train_heatmap_2classes来训练
class semantic_energy_loss(nn.Module):
    def __init__(self):
        super(semantic_energy_loss,self).__init__()
        self.init_protype=torch.abs(torch.randn(1, 1, 70))

        self.protype_tumor = nn.Parameter(self.init_protype, requires_grad=False)
        self.protype_tissue = nn.Parameter(self.init_protype, requires_grad=False)
    
    def forward(self,output_feature, nodes_label, regression, class_heat_maps,
                            lam1=2.0, lam2=0.05, lam3=0.2,lam_protype=0.2,momentum=0.01):
        # target是稀疏标记,用来计算energy
        # print(self.protype_tumor)
        # print(self.protype_tissue)

        B, dim, H, W = output_feature.shape
        nodes_label = nodes_label.reshape(B, -1)

        num_node = H * W
        output_feature = output_feature.reshape(B, dim, num_node)
        output_feature = output_feature.permute(0, 2, 1)

        # print(output_feature.shape)
        # # print(output_feature.shape)

        total_regression_energy = torch.zeros(B, 3, 32, 32).cuda()

        #肿瘤细胞
        mask1 = (nodes_label==1).cuda()
        mask1 = mask1.reshape(B, 1, num_node)
        num_instances1 = mask1.float().sum(dim=2).reshape(B, 1)
        num_instances1[num_instances1 == 0] = 1e-10
        # print(num_instances1)

        #其他组织细胞
        mask2 = (nodes_label==2).cuda()
        mask2 = mask2.reshape(B, 1, num_node)
        num_instances2 = mask2.float().sum(dim=2).reshape(B, 1)
        num_instances2[num_instances2 == 0] = 1e-10


        x1_normalized = output_feature / torch.norm(output_feature, dim=2, keepdim=True)
        x2_normalized = output_feature / torch.norm(output_feature, dim=2, keepdim=True)

        cos_distance = torch.einsum('ijk,ilk->ijl', x1_normalized, x2_normalized)
        cos_distance=(1+cos_distance)/2

        protype_tumor=self.protype_tumor.repeat(B,1,1).cuda()
        protype_tumor_normalize=protype_tumor/torch.norm(protype_tumor,dim=2,keepdim=True)

        protype_tissue=self.protype_tissue.repeat(B,1,1).cuda()
        protype_tissue_normalize=protype_tissue/torch.norm(protype_tissue,dim=2,keepdim=True)

        cos_distance_tumor = torch.einsum('ijk,ilk->ijl', x1_normalized, protype_tumor_normalize).reshape(B,num_node)
        cos_distance_tissue=torch.einsum('ijk,ilk->ijl', x1_normalized, protype_tissue_normalize).reshape(B,num_node)

        # cos_distance=cos_distance.reshape(B,num_node)

        regression_energy1=mask1*cos_distance
        regression_energy1=torch.sum(regression_energy1,dim=2)/num_instances1

        # lam_protype_tumor = torch.tensor(lam_protype, dtype=torch.float32)
        # lam_protype_tumor=lam_protype_tumor.repeat(B,1)
        mask=num_instances1==1e-10
        # print('-------------')
        # print(cos_distance_tumor)
        # print(regression_energy1)
        # regression_energy1=regression_energy1+1e-10
        regression_energy1=lam_protype*cos_distance_tumor+(1-lam_protype)*(regression_energy1*(~mask)+mask*cos_distance_tumor)

        # regression_energy1=lam_protype*cos_distance_tumor+(1-lam_protype)*(regression_energy1*(~mask)+mask*cos_distance_tumor)
        # print(regression_energy1)
        regression_energy1=1-regression_energy1

        mask=mask1+mask2
        num_instances=num_instances1+num_instances2
        regression_energy=mask*cos_distance
        regression_energy=torch.sum(regression_energy,dim=2)/num_instances
        regression_energy=1-regression_energy

        regression_energy1[nodes_label == 1] = 1#1
        regression_energy1[nodes_label == 2] = 1

        regression_energy[nodes_label == 1]=1
        regression_energy[nodes_label == 2]=1


        mask=num_instances2==1e-10
        regression_energy2=mask2*cos_distance
        regression_energy2=torch.sum(regression_energy2,dim=2)/num_instances2
        # regression_energy2=regression_energy2+1e-10
        regression_energy2=lam_protype*cos_distance_tissue+(1-lam_protype)*(regression_energy2*(~mask)+mask*cos_distance_tissue)
        regression_energy2=1-regression_energy2

        regression_energy2[nodes_label == 2] = 1#1
        regression_energy2[nodes_label == 1] = 1

        mask=mask1+mask2
        # num_instances=num_instances1+num_instances2

        #计算使标注点和未标注点的距离尽可能远
        stretched_feature_loss=0
        for batch in range(B):
            labeled_index1= mask[batch,:,:]==True   #标记点
            # labeled_index2= mask2[batch,:,:]==True   #标记点
            unlabeled_index= mask[batch,:,:]==False #未标记点
            # print(labeled_index.shape)

            labeled_feature1=output_feature[batch,labeled_index1[0,:],:]
            # labeled_feature2=output_feature[batch,labeled_index2[0,:],:]
            unlabeled_feature=output_feature[batch,unlabeled_index[0,:],:]

            feature_distance1=torch.cdist(labeled_feature1, unlabeled_feature, p=2)
            feature_distance1=torch.exp(-(feature_distance1-3))        
            feature_distance1=torch.mean(feature_distance1)

            # feature_distance2=torch.cdist(labeled_feature2, unlabeled_feature, p=2)
            # feature_distance2=torch.exp(-(feature_distance2-3))        
            # feature_distance2=torch.mean(feature_distance2)

            lam_labeled1=num_instances1[batch,:]+num_instances2[batch,:]/(num_node*0.05)
            # lam_labeled2=num_instances2[batch,:]/(num_node*0.05)

            stretched_feature_loss=stretched_feature_loss+lam_labeled1*feature_distance1#+lam_labeled2*feature_distance2
            

        stretched_feature_loss=stretched_feature_loss/B
        stretched_feature_loss=stretched_feature_loss.squeeze()

        # mask1 = (nodes_label==1).cuda()

        total_regression_energy[:, 1] =regression_energy1.reshape(B, H, W)
        total_regression_energy[:, 2] =regression_energy2.reshape(B, H, W)

        # mean_distance=torch.sum(distance,dim=1)
        # mean_distance=torch.sum(mean_distance,dim=1)/1024
        # mean_distance=torch.mean(distance,dim=1)
        # mean_distance=torch.mean(mean_distance,dim=1)
        # mean_distance=mean_distance.view(B,1,1)
        # regression_energy=torch.min(total_regression_energy[:, 1],total_regression_energy[:, 2])

        total_regression_energy[:, 0] =regression_energy.reshape(B, H, W)#((total_regression_energy[:, 1]+total_regression_energy[:, 2])/2).reshape(B, H, W)#mean_distance.expand(B,H,W)#torch.mean(distance,dim=0)#mean_diatance.reshape(B,H,W)#torch.mean(distance)#1e-6 #torch.mean(distance)#1e-7#0.0001#torch.mean(regression_energy)
        # total_regression_energy=2*F.softmax(total_regression_energy,dim=1)
        # total_regression_energy=F.normalize(total_regression_energy,dim=1)

        # total_regression_energy = torch.clip_(total_regression_energy, 0.0, 1.0)
        total_regression_energy = F.interpolate(total_regression_energy, scale_factor=32, mode='bilinear')

        class_heat_maps = class_heat_maps.type(torch.float32)

        regression_loss=-class_heat_maps*torch.log(torch.clip(regression,1e-10,1.0))
        regression_loss=regression_loss*total_regression_energy
        regression_loss = torch.sum(regression_loss, dim=1)
        regression_loss = torch.mean(regression_loss)

        ###hinge_loss###
        # hinge_loss=torch.mean((1-regression_energy)**2)

        # print(class_heat_maps.shape)
        labeled_region=class_heat_maps[:,1:2]+class_heat_maps[:,2:3]
        labeled_region[labeled_region>0] = 1

        # print(labeled_region.shape)
        labeled_pixels=torch.sum(labeled_region.reshape(B,-1),dim=1)
        # labeled_pixels=torch.sum(labeled_pixels,dim=1)
        # print(labeled_pixels)
        heatmap=class_heat_maps
        heatmap[:,1:]=(heatmap[:,1:]>0)
        heatmap[:,0:1]=1-heatmap[:,1:2]-heatmap[:,2:3]

        exclusion_loss=(1-heatmap)*torch.log1p(torch.clip(regression,1e-10,1.0))
        exclusion_loss=labeled_region*exclusion_loss

        exclusion_loss = torch.sum(exclusion_loss.reshape(B,-1),dim=1)/labeled_pixels+1e-10
        exclusion_loss= torch.mean(exclusion_loss).squeeze()

        # exclusion_loss=(1-heatmap)*torch.log(1+torch.clip(regression,1e-10,1.0))
        # exclusion_loss=labeled_region*exclusion_loss
        # # exclusion_loss = torch.sum(exclusion_loss, dim=1)
        # exclusion_loss = torch.sum(exclusion_loss.reshape(B,-1),dim=1)/labeled_pixels
        # exclusion_loss= torch.mean(exclusion_loss).squeeze()
        # print(exclusion_loss)

        # total_loss = lam1 * regression_loss + lam2 * stretched_feature_loss
        total_loss = lam1 * regression_loss + lam2 * stretched_feature_loss  + lam3 * exclusion_loss
        # print('--------------')
        # print(regression_loss)
        # print(stretched_feature_loss)
        # print(exclusion_loss)

        mask=(num_instances1>1e-10)
        mean_tumor_feature=torch.sum(mask1.permute(0,2,1)*output_feature,dim=1,keepdim=True)
        mean_tumor_feature=mean_tumor_feature/num_instances1.unsqueeze(2).expand(B, 1, 70)
        mean_tumor_feature=torch.sum(mean_tumor_feature,dim=0,keepdim=True)/(torch.sum(mask)+1e-10)
        protype_tumor=momentum*mean_tumor_feature+(1-momentum)*protype_tumor
        # batch_num=torch.sum(num_instances1>1e-10)
        # protype_tumor=torch.sum(protype_tumor,dim=0,keepdim=True)/(batch_num+1e-10)
        protype_tumor=torch.mean(protype_tumor,dim=0,keepdim=True)
        # print("-------------")
        # print(protype_tumor)
        # print(mean_tumor_feature)


        # print(mean_tumor_feature)
        
        self.protype_tumor=nn.Parameter(protype_tumor)

        mask=(num_instances2>1e-10)
        mean_tissue_feature=torch.sum(mask2.permute(0,2,1)*output_feature,dim=1,keepdim=True)
        mean_tissue_feature=mean_tissue_feature/num_instances2.unsqueeze(2).expand(B, 1, 70)
        mean_tissue_feature=torch.sum(mean_tissue_feature,dim=0,keepdim=True)/(torch.sum(mask)+1e-10)
        protype_tissue=momentum*mean_tissue_feature+(1-momentum)*protype_tissue
        # batch_num=torch.sum(num_instances2>1e-10)
        protype_tissue=torch.mean(protype_tissue,dim=0,keepdim=True)#/(batch_num+1e-10)
        

        self.protype_tissue=nn.Parameter(protype_tissue)


        # return regression_loss
        return total_loss, regression_loss, stretched_feature_loss, exclusion_loss