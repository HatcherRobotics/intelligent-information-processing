import torch
import torch.nn as nn
class My_SoftmaxNet(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.w = nn.Parameter(torch.normal(0, 0.01, size=(self.num_inputs, self.num_outputs), requires_grad=True))
        self.b = nn.Parameter(torch.zeros(self.num_outputs, requires_grad=True))
        self.sigmoid = nn.Sigmoid()
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def softmax(self,x):
        X_exp = torch.exp(x)
        partition = X_exp.sum(1, keepdim=True)
        return X_exp / partition  # 这里应用了广播机制
    
    def forward(self,x):
        initial_output = self.sigmoid(torch.mm(x.view(-1,self.num_inputs),self.w)+self.b)
        softmax_output = self.softmax(initial_output)
        return softmax_output