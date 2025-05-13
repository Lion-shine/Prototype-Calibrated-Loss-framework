import os
import cv2
import h5py
import json
import numpy as np
import math
import random
import torch

# 指定总的大文件夹路径
root_dir = '/home/aaa/disk1/vig_pytorch/ori_data'

cnt=0
img_paths=[]

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
        heatmap += generate_heatmap(x, y, radius, width, height)  #可视化要修改坐标位置
    return heatmap

def visiulization_heatmap(img_path,vis_heatmap,points,remark):
    img=cv2.imread(img_path)
    vis_heatmap=cv2.applyColorMap((vis_heatmap*255).astype(np.uint8),cv2.COLORMAP_JET)
    for point,label in zip(points,remark):
        x, y = map(int, point)
        color=colors[label]
        cv2.circle(img, (x, y), 3, color, -1)

    cv2.imshow('heatmap', vis_heatmap)
    cv2.imshow('img',img)
    cv2.waitKey(0)


def generate_random_number():
    """生成符合非欧几里得分布的0-1之间的随机数"""
    return math.log(1-random.uniform(0, 1))/(-1)


def read_json_file(file_name,label_dict,store_name):
    # 打开 JSON 文件并将其转换为 Python 对象
    with open(file_name) as f:
        data = json.load(f)

    # 提取出roilist
    roilist = data['roilist']
    
    # 遍历roilist，提取jgResult和path中的x和y
    coords = []
    remark = []
    text_cnt=0
    for roi in roilist:
        try:
            label=roi['remark']
            if label in label_dict:
                label=label_dict[label]
                remark.append(label)
                x_values = roi['path']['x']
                y_values = roi['path']['y']
                if x_values[0] >= 1024 : print("find it")
                if y_values[0] >= 1024 :print("find it")
                coords.append((x_values[0], y_values[0]))

                
            else:
                print('cannot find this label in label_dict:  '+str(label))
                print(roi)
        except KeyError:
            print("cannot find it")
    
    sparse_class_heat_map,sparse_position_heat_map,sparse_coords,sparse_coords_label,class_heat_map,position_heat_map,target=generate_class_and_position_heatmap(coords,remark)
    
    ###   可视化   ###
    # visiulization_heatmap(img_path='/Volumes/KINGSTON/ori_data/20210514 her2 导出数据/001  1+/2021_01_15_20_12_02_757900/319187/9.png',vis_heatmap=class_heat_map[1],points=sparse_coords,remark=sparse_coords_label)
    # print(sparse_class_heat_map[1].shape)

    if(len(coords)!=len(remark)):
        print(file_name)
        print(len(coords))
        print(len(remark))
        print(text_cnt)



    # 将数据存储到HDF5文件中
    with h5py.File(store_name, 'w') as f:
        # 将jgPesult写入HDF5文件
        f.create_dataset('sparse_target', data=target)  #存储稀疏标注区域，在相应的标注区域打label  (N,N),用于制作semantic energy

        # 存储稀疏热图
        f.create_dataset('sparse_class_heat_map', data=sparse_class_heat_map)
        f.create_dataset('sparse_position_heat_map', data=sparse_position_heat_map)

        #存储所有点坐标热图
        f.create_dataset('class_heat_map', data=class_heat_map)
        f.create_dataset('position_heat_map', data=position_heat_map)



        f.create_dataset('sparse_coords', data=sparse_coords)   #存储所有稀疏点坐标
        f.create_dataset('sparse_coords_label', data=sparse_coords_label)   #存储稀疏点对应的label
        
        # 将coords写入HDF5文件
        coords_as_array = [list(coord) for coord in coords]
        f.create_dataset('coords', data=coords_as_array)   #存储所有标注坐标--->后面要换成（N，N）的形式

def generate_class_and_position_heatmap(coords,remark,H=1024,W=1024,n_class=10,radius=15,H_num=32,W_num=32):
    # H_num代表着横排有多少个patch
    # H_dim代表着一个patch在横排维度是多少
    
    sparse_coords = []
    sparse_coords_label=[]
    sparse_class_heat_map=np.zeros((n_class,H,W))
    class_heat_map=np.zeros((n_class,H,W))
    sparse_position_heat_map=np.zeros((2,H,W))
    position_heat_map=np.zeros((2,H,W))
    label_num=np.zeros((n_class))
    target=np.zeros((H_num,W_num))
    # text_cnt=0
    
    for i in range(len(coords)):
        coord=coords[i]
        label=remark[i]
        x=int(coord[0])
        y=int(coord[1])
        
        # print(coord)
        # print(str(x)+" "+str(y))
        if x<1024 and y<1024:
            label_num[label]+=1

            #保证该类至少有一个标注点
            if label_num[label]>1:
                rand=generate_random_number()
                if rand<=0.1:                       #调节稀疏点数量
                    # text_cnt+=1
                    target[int(x/H_num),int(y/W_num)]=label
                    sparse_coords.append((coord[0],coord[1]))
                    sparse_coords_label.append(label)
            elif label_num[label]==1:
                # text_cnt+=1
                target[int(x/H_num),int(y/W_num)]=label
                sparse_coords.append((coord[0],coord[1]))
                sparse_coords_label.append(label)
        
    # print(text_cnt)
    # print(len(coords))
    
    
    for c in range(n_class):
        sparse_class_points=[point for i,point in enumerate(sparse_coords) if sparse_coords_label[i]==c ]
        class_points=[point for i,point in enumerate(coords) if remark[i]==c ]
        sparse_class_heat_map[c] = generate_gaussian_heatmap(sparse_class_points, radius, W, H)
        class_heat_map[c] = generate_gaussian_heatmap(class_points, radius, W, H)

    sparse_class_points=np.clip(sparse_class_points,0.0,1.0)
    class_points=np.clip(class_points,0.0,1.0)
    sparse_class_heat_map=np.clip(sparse_class_heat_map,0.0,1.0)
    class_heat_map=np.clip(class_heat_map,0.0,1.0)
    
    sum_obj_sparse_heatmap=np.sum(sparse_class_heat_map[1:10],axis=0)
    sparse_class_heat_map[0]=1-sum_obj_sparse_heatmap
    sparse_class_heat_map[0]=np.clip(sparse_class_heat_map[0],0.0,1.0)

    sum_obj_heatmap=np.sum(class_heat_map[1:10],axis=0)
    class_heat_map[0]=1-sum_obj_heatmap
    class_heat_map[0]=np.clip(class_heat_map[0],0.0,1.0)

    sparse_position_heat_map[1]=generate_gaussian_heatmap(sparse_coords, radius, W, H)
    sparse_position_heat_map[0]=1-sparse_position_heat_map[1]
    sparse_position_heat_map[0]=np.clip(sparse_position_heat_map[0],0.0,1.0)

    position_heat_map[1]=generate_gaussian_heatmap(coords, radius, W, H)
    position_heat_map[0]=1-position_heat_map[1]
    position_heat_map[0]=np.clip(position_heat_map[0],0.0,1.0)


    return sparse_class_heat_map,sparse_position_heat_map,sparse_coords,sparse_coords_label,class_heat_map,position_heat_map,target


colors = [
    (255, 0, 0),  # 红色
    (0, 255, 0),  # 绿色
    (0, 0, 255),  # 蓝色
    (255, 255, 0),  # 黄色
    (255, 0, 255),  # 洋红色
    (0, 255, 255),  # 青色
    (128, 0, 0),  # 深红色
    (0, 128, 0),  # 深绿色
    (0, 0, 128),  # 深蓝色
    (128, 128, 0),  # 深黄色
]

label_dict={
    '阴性肿瘤细胞':1,
    '纤维细胞':2,
    '淋巴细胞':3,
    '血管内皮细胞':4,
    '难以区分的非肿瘤细胞':5,
    '组织细胞':6,
    '脂肪细胞':7,
    '导管上皮细胞':8,
    
    '微弱的不完整膜阳性肿瘤细胞':9,
    '弱-中等的完整细胞膜阳性肿瘤细胞':9,
    '强度的完整细胞膜阳性肿瘤细胞':9,
    '中-强度的不完整细胞膜阳性肿瘤细胞':9,
    '阳性破损肿瘤细胞':9,
    '阳性完整膜轻度至中度肿瘤细胞':9,
    '阳性完整膜重度肿瘤细胞':9,

    #'unlabeled':0,
    }

label_dict={
    '阴性肿瘤细胞':2,
    '纤维细胞':3,
    '淋巴细胞':3,
    '血管内皮细胞':3,
    '难以区分的非肿瘤细胞':2,
    '组织细胞':3,
    '脂肪细胞':3,
    '导管上皮细胞':3,
    
    '微弱的不完整膜阳性肿瘤细胞':1,
    '弱-中等的完整细胞膜阳性肿瘤细胞':1,
    '强度的完整细胞膜阳性肿瘤细胞':1,
    '中-强度的不完整细胞膜阳性肿瘤细胞':1,
    '阳性破损肿瘤细胞':1,
    '阳性完整膜轻度至中度肿瘤细胞':1,
    '阳性完整膜重度肿瘤细胞':1,

    #'unlabeled':0,
    }


out_put_path=os.path.join('/Users/xuzhengyang/Documents/实验/code测试',)
# read_json_file('/Volumes/KINGSTON/ori_data/20210514 her2 导出数据/001  1+/2021_01_15_20_12_02_757900/319187/index.json',label_dict,'/Users/xuzhengyang/Documents/实验/code测试/text.h5')

# 遍历总的大文件夹中的每个大文件夹
for folder1 in os.listdir(root_dir):
    folder1_path = os.path.join(root_dir, folder1)

    # 如果当前项不是文件夹，则跳过
    if not os.path.isdir(folder1_path):
        continue
    
    for folder2 in os.listdir(folder1_path):
        folder2_path=os.path.join(folder1_path,folder2)

        # 如果当前项不是文件夹，则跳过
        if not os.path.isdir(folder2_path):
            continue    

        for folder in os.listdir(folder2_path):
            folder3_path = os.path.join(folder2_path, folder)

            # 如果当前项不是文件夹，则跳过
            if not os.path.isdir(folder3_path):
                continue

            # 遍历当前大文件夹中的每个存储数据的文件夹
            for data_folder in os.listdir(folder3_path):
                data_folder_path = os.path.join(folder3_path, data_folder)

                # 如果当前项不是文件夹，则跳过
                if not os.path.isdir(data_folder_path):
                    continue

                # 遍历当前存储数据的文件夹中的每个文件
                for file in os.listdir(data_folder_path):
                    file_path = os.path.join(data_folder_path, file)

                    # 如果当前文件不是图片或者 json 文件，则跳过
                    if not (file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png")  or file.endswith(".json")):
                        print('cannot find the data')
                        continue

                    # 如果当前文件是图片文件，则处理图片
                    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                        if (file.startswith('._') and not file.startswith(".__")) or file.startswith('thumbnail'):
                            continue
                        else:
                            cnt+=1
                            
                            img_path=os.path.join(root_dir,folder1,folder2,folder,data_folder,file)
                            # print(img_path)
                            img=cv2.imread(img_path)
                            H,W,_=img.shape

                            if H==1024 and W==1024:
                                #防止产生重复标号
                                new_name=str(cnt)+".png"
                                # img_path_new_name=img_path.replace(file,new_name)
                                save_img_path = os.path.join('/home/aaa/disk1/vig_pytorch/data/imgve', new_name)
                                # print(save_img_path)
                                cv2.imwrite(save_img_path,img)

                                json_path=img_path.replace(file,'index.json')
                                h5_output_path = os.path.join('/home/aaa/disk1/vig_pytorch/data/ground_truth',
                                                              new_name).replace('.png', '.h5')
                                # print(h5_output_path)
                                read_json_file(json_path,label_dict,h5_output_path)


                            # print(file.replace('.png','.json'))

                            # json_file=os
                            # print(img_path)
                            # os.rename(file,str(cnt)+".png")
                            # print(file)
                            
                            
                        

                    # 如果当前文件是 json 文件，则处理 json 文件
                    # if file.endswith(".json"):
                    #     # 处理 json 文件的代码
                    #     print("处理 json 文件：" + file_path)

print("总的图片数目:   "+str(cnt))
# print(img_paths)

