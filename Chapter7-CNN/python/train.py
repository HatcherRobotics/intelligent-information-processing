from dataset import *
from resnet import *
from utils import *
import torch.optim as optim
num_epochs=100
lr=0.01
loss=nn.CrossEntropyLoss()

net = ResNet()
net.to(device)
optimizer=optim.Adam(net.parameters(),lr,weight_decay=1e-5)
for params in net.parameters():
    nn.init.normal_(params,mean=0,std=0.01)

train_loss,val_loss,train_accuracy,val_accuracy=train(net,train_iter,val_iter,loss,num_epochs,optimizer)
visualization(train_loss,val_loss,train_accuracy,val_accuracy)