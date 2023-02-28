import math
import numpy as np
#RMSE均方误差
#MAE平均绝对误差
#MAPE平均绝对百分比误差
def mse_fn(y_true,y):
    mse = (y_true-y)**2
    return mse

def mae_fn(y_true,y):
    mae = math.fabs(y_true-y)
    return mae

def mape_fn(y_true,y):
    mape = math.fabs((y_true-y)/y_true)
    mape = mape*100
    return mape

def eval(y_true,y):
    mse = mse_fn(y_true,y)
    rmse = math.sqrt(mse)
    mae = mae_fn(y_true,y)
    mape = mape_fn(y_true,y)
    return [rmse, mae, mape]

def error_quantification(l):
    l = np.array(l)
    sum=0
    for i in range(len(l)):
        sum = sum+(l[i,0])**2
    RMSE = sum/len(l)
    sum=0
    for i in range(len(l)):
        sum = sum+l[i,1]
    MAE = sum/len(l)
    sum=0
    for i in range(len(l)):
        sum = sum+l[i,2]
    MAPE = sum/len(l)
    return [RMSE,MAE,MAPE]    

