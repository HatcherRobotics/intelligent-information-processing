import argparse
import torch.optim as optim
from dataset import *
from SoftmaxNet import *
from utils import *
from visualization import *

parser = argparse.ArgumentParser(description='Softmax Regression')
parser.add_argument('--num_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--lr', type=float, default=0.3, help='optimizer learning rate')
parser.add_argument('--num_inputs',type=int,default=784,help='number of dimensions of input data')
parser.add_argument('--num_outputs',type=int,default=10,help='number of dimensions of output data')
parser.add_argument('--weight_decay',type=float,default=1e-3,help='optimizer weight decay')

args = parser.parse_args()

net = My_SoftmaxNet(args.num_inputs,args.num_outputs)
net.to(device)
optimizer_w = optim.SGD(params=[net.w],lr=args.lr,weight_decay=args.weight_decay)
optimizer_b = optim.SGD(params=[net.b],lr=args.lr)
loss = cross_entropy

train_loss,test_loss,train_accuracy,test_accuracy=train(net,train_iter,test_iter,loss,args.num_epochs,optimizer_w,optimizer_b)
visualization(train_loss,test_loss,train_accuracy,test_accuracy)