from resnet import *
device = torch.device("cuda")
net = ResNet()
net.to(device)

for name,parameters in net.named_parameters():
    print(name,':',parameters.size())
#参数量
from torchsummary import summary
summary(net, input_size=(3,256,256))

#计算量
from ptflops import get_model_complexity_info
macs, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=False, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#显存占用量
print("显存占用量为：",torch.cuda.memory_allocated()/(1024*1024),"MB")
#到调用函数为止所达到的最大的显存占用字节数
print("到调用函数为止所达到的最大的显存占用字节数:",torch.cuda.max_memory_allocated()/(1024*1024),"MB")