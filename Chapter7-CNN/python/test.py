import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

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
            if(img is None):
                print(pic)
            try:
                img = cv.resize(img, (32, 32))
            except:
                print(pic)
            img = np.array(img)
            img = img/255
            pics_list.append(img)
    x = torch.Tensor(np.array(pics_list)).permute(0,3,1,2)
    return x

#train_people_x = dataset_x('/home/hatcher/test/intelligent-information-processing/Chapter7-CNN/CTH/train/people')
test_empty_x = dataset_x('/home/hatcher/test/intelligent-information-processing/Chapter7-CNN/CTH/test/empty')