import pandas as pd
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
import numpy as np

df = pd.read_excel('CTG.xls','Data')
df.fillna(0,inplace=True)

features = np.array(df.loc[1:2126,1:21])
target = np.array(df.loc[1:2126,22]).astype('int')

x_train,x_test,y_train,y_test = model_selection.train_test_split(features,target,random_state=1,test_size=0.2)

classifier = svm.SVC(kernel='linear',gamma=0.1,decision_function_shape='ovo',C=0.1)
classifier.fit(x_train,y_train)
print("SVM-输出训练集的准确率为：",classifier.score(x_train,y_train))
print("SVM-输出测试集的准确率为：",classifier.score(x_test,y_test))
y_hat = classifier.predict(x_test)
classreport = metrics.classification_report(y_test,y_hat)
print(classreport)