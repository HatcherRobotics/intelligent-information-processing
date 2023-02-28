from SensorFusion import *
from error import *
def simulation(n):
    y_list = []
    y_true_list=[]
    error_list = []
    for i in range(n):
        y_true = random.randint(1,100)
        y_true_list.append(y_true)
        sigma1 = random.random()
        sigma2 = random.random()
        sigma3 = random.random()
        y = WeightFusion(y_true,sigma1,sigma2,sigma3)
        y_list.append(y)
        error = eval(y_true,y)
        error_list.append(error)
    return y_true_list,y_list,error_list