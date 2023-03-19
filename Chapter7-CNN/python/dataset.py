import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
def dataset_x(path,data_augmentaion=False):
    pics = os.listdir(path)
    pics_list=[]
    if(data_augmentaion==True):
        for pic in pics:
            img = cv.imread(path+'/'+pic)
            #img = cv.resize(img, (32, 32))
            img = np.array(img)
            img = img/255
            flip_img = np.array(cv.flip(img, 1))
            flip_img=flip_img/255
            equ_img = cv.equalizeHist(cv.cvtColor(img,cv.COLOR_BGR2GRAY))
            equ_img=np.array(cv.cvtColor(equ_img,cv.COLOR_GRAY2BGR))
            equ_img=equ_img/255
            pics_list.append(img)
            pics_list.append(flip_img)
            pics_list.append(equ_img)
    else:
        for pic in pics:
            img = cv.imread(path+'/'+pic)
            img = cv.resize(img, (32, 32))
            img = np.array(img)
            img = img/255
            pics_list.append(img)
    x = torch.Tensor(np.array(pics_list)).permute(0,3,1,2)
    return x

def dataset_y(dataset_x,type):
    size = dataset_x.shape[0]
    y = type*torch.ones([size])
    return y

train_empty_x = dataset_x('/home/hatcher/test/intelligent-information-processing/Chapter7-CNN/CTH/train/empty')
train_people_x = dataset_x('/home/hatcher/test/intelligent-information-processing/Chapter7-CNN/CTH/train/people')
train_train_x = dataset_x('/home/hatcher/test/intelligent-information-processing/Chapter7-CNN/CTH/train/train')
val_empty_x = dataset_x('/home/hatcher/test/intelligent-information-processing/Chapter7-CNN/CTH/val/empty')
val_people_x = dataset_x('/home/hatcher/test/intelligent-information-processing/Chapter7-CNN/CTH/val/people')
val_train_x = dataset_x('/home/hatcher/test/intelligent-information-processing/Chapter7-CNN/CTH/val/train')
test_empty_x = dataset_x('/home/hatcher/test/intelligent-information-processing/Chapter7-CNN/CTH/test/empty')
test_people_x = dataset_x('/home/hatcher/test/intelligent-information-processing/Chapter7-CNN/CTH/test/people')
test_train_x = dataset_x('/home/hatcher/test/intelligent-information-processing/Chapter7-CNN/CTH/test/train')
train_empty_y = dataset_y(train_empty_x,0)
train_people_y = dataset_y(train_people_x,1)
train_train_y = dataset_y(train_train_x,2)
val_empty_y = dataset_y(val_empty_x,0)
val_people_y = dataset_y(val_people_x,1)
val_train_y = dataset_y(val_train_x,2)
test_empty_y = dataset_y(test_empty_x,0)
test_people_y = dataset_y(test_people_x,1)
test_train_y = dataset_y(test_train_x,2)

batch_size = 48# 将训练数据的特征和标签组合
train_dataset = Data.TensorDataset(train_empty_x,train_empty_y)+Data.TensorDataset(train_people_x,train_people_y)+Data.TensorDataset(train_train_x,train_train_y)
val_dataset = Data.TensorDataset(val_empty_x,val_empty_y)+Data.TensorDataset(val_train_x,val_train_y)+Data.TensorDataset(val_people_x,val_people_y)
test_dataset = Data.TensorDataset(test_empty_x,test_empty_y)+Data.TensorDataset(test_people_x,test_people_y)+Data.TensorDataset(test_train_x,test_train_y)
# 把 dataset 放入 DataLoader
train_iter = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=2)
val_iter = Data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,num_workers=2)
test_iter = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,num_workers=2)