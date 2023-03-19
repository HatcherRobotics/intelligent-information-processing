from dataset import test_iter
from resnet import *
from utils import evaluate_accuracy
loss=nn.CrossEntropyLoss()
device = torch.device("cuda")
net = ResNet()
net.to(device)

net.load_state_dict(torch.load("/home/hatcher/test/intelligent-information-processing/Chapter7-CNN/python/resnet.pth"))
test_acc,_ = evaluate_accuracy(test_iter,net,loss)
print("在测试集上的准确率为:%.3f"%(test_acc))
