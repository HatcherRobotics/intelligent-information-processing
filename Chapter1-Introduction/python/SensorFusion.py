import random
def GenerateObservation(y_true,sigma):
    x = random.normalvariate(y_true,sigma)
    return x

def CalculateWeight(sigma1,sigma2,sigma3):
    w1 = (1/sigma1**2)/((1/sigma1**2)+(1/sigma2**2)+(1/sigma3**2))
    return w1

def WeightFusion(y_true,sigma1,sigma2,sigma3):
    y1 = GenerateObservation(y_true,sigma1)
    w1 = CalculateWeight(sigma1,sigma2,sigma3)
    y2 = GenerateObservation(y_true,sigma2)
    w2 = CalculateWeight(sigma2,sigma1,sigma3)
    y3 = GenerateObservation(y_true,sigma3)
    w3 = CalculateWeight(sigma3,sigma1,sigma2)
    y = w1*y1+w2*y2+w3*y3
    return y