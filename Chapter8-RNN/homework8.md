> 范峻铭22121286
>
> https://github.com/HatcherRobotics/intelligent-information-processing
>
> 代码均为原创，雷同即为抄袭

#### 数据清洗与数据加载器

使用Pandas读取Excel中的数据，去掉五组气压相关的数据，对其余数据进行归一化处理，十四维数据作为特征，以以为一维数据(极大风速)为标签；

```python
    def __init__(self,predict_step=3,label=1):
        self.raw_data = np.array(pd.read_excel("weather.xlsx")).astype(float)
        # 数据标准化
        self.predict_step = predict_step
        self.label = label
        self.min = np.amin(self.raw_data)
        self.max = np.amax(self.raw_data)
        self.data = (self.raw_data - self.min) / (self.max - self.min)

```

反归一化函数在评估模型以及数据可视化中用到，将归一化数据变换为原始数据的尺度；

```python
    def denormalize(self, x):
        return x * (self.max - self.min) + self.min
```

采用滑动窗口的方式建立数据集，步长为1，序列长度为50，预测布长为3；

```python
    def construct_set(self, train_por=0.65,val_por=0.2,test_por=0.15, window_size=50):
        ......
        for i in range(train_seqs.shape[0] - window_size-self.predict_step+1):
            train_seq = train_seqs[i:i+window_size+self.predict_step]
            train_x.append(train_seq[0:window_size,:])
            train_y.append(train_seq[window_size:window_size+self.predict_step,self.label])

        for i in range(val_seqs.shape[0] - window_size-self.predict_step+1):
            val_seq = val_seqs[i:i+window_size+self.predict_step]
            val_x.append(val_seq[0:window_size,:])
            val_y.append(val_seq[window_size:window_size+self.predict_step,self.label])

        for i in range(test_seqs.shape[0] - window_size-self.predict_step+1):
            test_seq = test_seqs[i:i+window_size+self.predict_step]
            test_x.append(test_seq[0:window_size,:])
            test_y.append(test_seq[window_size:window_size+self.predict_step,self.label])
        ......
```

#### LSTM:长短期记忆网络

##### 搭建神经网络

调库实现LSTM，取其最后一步的输出，将其通过两层全连接以及激活函数得到3维输出，对应三个小时的预测值；

```python
class LSTMNet(nn.Module):
    def __init__(self,batch_size,input_size,hidden_size,output_size,num_layers,seq_len,device="cuda"):
        super().__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=False)
        self.outlinear = nn.Sequential(
            nn.Linear(in_features=self.hidden_size,out_features=hidden_size//2),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size//2,out_features=self.output_size)
        )
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)        
    def forward(self,x):
        h,c = (torch.zeros(self.num_layers,self.seq_len,self.hidden_size).to(x.device) for _ in range(2))
        H,(h,c) = self.lstm(x,(h,c))
        out = H[:,-1,:].squeeze()
        out = self.outlinear(out)
        return out
```

##### 模型训练

取学习率为0.0001，优化器选择Adam，训练轮次为80轮，批量大小为16，LSTM的层数为3层。训练结果如下图所示

<img src="/run/user/1000/doc/af8057b2/lstm_loss.png" alt="lstm_loss" style="zoom:25%;" />

##### 模型评估

使用平均均方根误差RMSE，平均绝对误差MAE，平均绝对百分比误差MAPE来评估模型：

<img src="/run/user/1000/doc/431b7d2/lstm_eval.png" alt="lstm_eval" style="zoom: 25%;" />

##### 数据可视化

1h,2h,3h的预测值与真实值在训练集比较如下图所示

<img src="/run/user/1000/doc/47c2838e/lstm_train_1h.png" style="zoom:25%;" />

<img src="/run/user/1000/doc/b6961e57/lstm_train_2h.png" style="zoom:25%;" />

<img src="/run/user/1000/doc/b7f30b74/lstm_train_3h.png" style="zoom:25%;" />

1h,2h,3h的预测值与真实值在验证集上的比较如下图所示

<img src="/run/user/1000/doc/e1462015/lstm_val_1h.png" style="zoom:25%;" />

<img src="/run/user/1000/doc/e1ef233/lstm_val_2h.png" style="zoom:25%;" />

<img src="/run/user/1000/doc/875f11ae/lstm_val_3h.png" style="zoom:25%;" />

1h,2h,3h的预测值与真实值在测试集上的比较如下图所示

<img src="/run/user/1000/doc/751a3841/lstm_test_1h.png" style="zoom:25%;" />

<img src="/run/user/1000/doc/7666f08a/lstm_test_2h.png" style="zoom:25%;" />

<img src="/run/user/1000/doc/3d4a9753/lstm_test_3h.png" style="zoom:25%;" />

#### TCN：时间卷积网络

考虑到非因果的普通一维卷积会涉及到未来信息，选择以空洞因果卷积为基础的时间卷积网络。

##### 搭建神经网络

原作者已开源代码，但代码为本人阅读论文后原创;

padding为了使输入序列与输出序列相等，使用Chomp1d去掉由最右端填充元素参与卷积的生成元素，确保卷积不使用未来特征。

```python
class Chomp1d(nn.Module):
    def __init__(self,chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = int(chomp_size)
    def forward(self,x):
        return x[:,:,0:-self.chomp_size].contiguous()
#在padding与chomp浪费了大量时间为了使输出序列相等
class TCNResidualBlock(nn.Module):
    def __init__(self,inchannel,outchannel,kernelsize,dilation):
        super(TCNResidualBlock,self).__init__()
        self.ke = int(kernelsize+(kernelsize-1)*(dilation-1))
        self.conv = nn.Sequential(
            weight_norm(nn.Conv1d(inchannel,outchannel,kernel_size=kernelsize,dilation=dilation,padding=self.ke-1)),
            Chomp1d(self.ke-1),    
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            weight_norm(nn.Conv1d(outchannel,outchannel,kernel_size=kernelsize,dilation=dilation,padding=self.ke-1)),
            Chomp1d(self.ke-1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.conv1x1 = nn.Sequential( 
            weight_norm(nn.Conv1d(inchannel,outchannel,kernel_size=1)),
            )
        
    def forward(self,X):
        Y = self.conv(X)
        X = self.conv1x1(X)
        out = F.relu(X+Y)
        return out
```

```python
#通道数是特征的意思
#input: batch_size * features * seq_length但数据集的输入是batch_size * seq_length * features
block1 = TCNResidualBlock(14,32,5,1)
block2 = TCNResidualBlock(32,16,5,2)
block3 = TCNResidualBlock(16,8,3,4)

class TCNet(nn.Module):
    def __init__(self,hidden_size,predict_step):
        super(TCNet,self).__init__()
        self.features = nn.Sequential(
            block1,block2,block3
        )
        self.flatten = nn.Flatten()
        self.regression = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, predict_step),
            nn.LeakyReLU()
        )
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    def forward(self,X):
        X = X.permute(0,2,1).contiguous()
        features = self.features(X)
        out = self.flatten(features)
        out = self.regression(out).squeeze()
        return out
```

##### 模型训练

取学习率为0.001，优化器选择Adam，训练轮次为80轮，批量大小为16，损失函数选择均方损失，堆叠三个TCN块，其空洞系数分别为1，2，4，卷积核的大小分别为5，5，3，输出通道数分别为32，16，8，训练结果如下图所示

<img src="/run/user/1000/doc/2445fb3a/tcn_loss.png" style="zoom:25%;" />

##### 模型评估

使用平均均方根误差RMSE，平均绝对误差MAE，平均绝对百分比误差MAPE来评估模型：

<img src="/run/user/1000/doc/ffd4a9b4/tcn_eval.png" style="zoom:25%;" />

##### 数据可视化

1h,2h,3h的预测值与真实值在训练集比较如下图所示

<img src="/run/user/1000/doc/6aa51b1/tcn_train_1h.png" style="zoom:25%;" />

<img src="/run/user/1000/doc/7cbf15f1/tcn_train_2h.png" alt="tcn_train_2h" style="zoom:25%;" />

<img src="/run/user/1000/doc/56416dbe/tcn_train_3h.png" style="zoom:25%;" />

1h,2h,3h的预测值与真实值在验证集比较如下图所示

<img src="/run/user/1000/doc/ebe90212/tcn_val_1h.png" alt="tcn_val_1h" style="zoom:25%;" />

<img src="/run/user/1000/doc/22220c5b/tcn_val_2h.png" style="zoom:25%;" />

<img src="/run/user/1000/doc/4d1928c1/tcn_val_3h.png" style="zoom:25%;" />

1h,2h,3h的预测值与真实值在测试集比较如下图所示

<img src="/run/user/1000/doc/f27832a/tcn_test_1h.png" alt="tcn_test_1h" style="zoom:25%;" />

<img src="/run/user/1000/doc/baadc434/tcn_test_2h.png" style="zoom:25%;" />

<img src="/run/user/1000/doc/5d4122d2/tcn_test_3h.png" style="zoom:25%;" />

#### SVM：支持向量机

核函数选择径向基函数：

```python
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
```

预测函数如下所示,依然使用滑动窗口法，窗口大小为50，步长为1，原数据、预测步长、评估方法、可视化策略等都与神经网络相同。

```python
def predict(x,y,window_size,predict_step,func,eval,visulization):
    predict = []
    test = []
    for i in range(x.shape[0]-window_size-predict_step+1):
        temp_train_x = raw_data[i:i+window_size,:]
        temp_train_y = y[i:i+window_size]
        temp_test_x = raw_data[i+window_size:i+window_size+predict_step,:]
        temp_test_y = y[i+window_size:i+window_size+predict_step]
        temp_predict = func.fit(temp_train_x,temp_train_y.reshape(-1)).predict(temp_test_x)
        predict.append(temp_predict)
        test.append(temp_test_y)
    predict = np.array(predict).squeeze()
    test = np.array(test).squeeze()
    print(predict.shape)
    print(test.shape)
    [rmse, mae, mape] = eval(test,predict)
    print("rmse:",rmse,"mae:",mae,"mape:",mape)
    visulization(predict,test)
predict(x,y,50,3,svr_rbf,eval,visulization)
```

结果如图：

<img src="/run/user/1000/doc/6b685f6/svm_1h.png" alt="svm_1h" style="zoom:25%;" />

<img src="/run/user/1000/doc/66d4e48e/svm_2h.png" style="zoom:25%;" />

<img src="/run/user/1000/doc/85a374d4/svm_3h.png" style="zoom:25%;" />

三种方法对比

|      | RMSE                   | MAE                  | MAPE                   |
| ---- | ---------------------- | -------------------- | ---------------------- |
| LSTM | 3.402497770937961      | 2.921602964401245    | **36.95100545883179%** |
| TCN  | 5.746902294830505      | 5.355382919311523    | 70.02002596855164%     |
| SVM  | **2.6913743661902703** | **2.00855275156032** | 56.84138900980481%     |

神经网络存在很大问题，虽然在测试集上能反映出大致的变换趋势，但在具体数值上存在较大误差；

TCN在验证集上的表现强于LSTM，但在测试集上不如LSTM，可能出现了过拟合。

SVM纯粹从数学角度进行拟合，效果稳定。