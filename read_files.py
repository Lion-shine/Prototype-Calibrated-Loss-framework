import os

# 指定总的大文件夹路径
root_dir = '/home/aaa/disk1/vig_pytorch/ori_data'

cnt=0
img_paths=[]

# 遍历总的大文件夹中的每个大文件夹
for folder1 in os.listdir(root_dir):
    folder1_path = os.path.join(root_dir, folder1)
    print(folder1_path)

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
                            # index,extension=os.path.splitext(file)
                            img_path=os.path.join(folder1,folder2,folder,data_folder,file)
                            img_paths.append(img_path)
                            json_path=img_path.replace(file,'index.json')
                            print(json_path)

                            # json_file=os
                            # print(img_path)
                            # os.rename(file,str(cnt)+".png")
                            print(file)
                            
                            
                        

                    # 如果当前文件是 json 文件，则处理 json 文件
                    # if file.endswith(".json"):
                    #     # 处理 json 文件的代码
                    #     print("处理 json 文件：" + file_path)

print("总的图片数目:   "+str(cnt))
# print(img_paths)