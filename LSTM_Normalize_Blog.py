import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import  pandas as pd
import  os
from keras.models import Sequential, load_model


def Create_dataset(dataset,look_back):
    data_X, data_Y = [], []
    for i in range(len(dataset) - look_back - 1 ):
        a = dataset[i:(i + look_back)]
        data_X.append(a)
        data_Y.append(dataset[i + look_back])
    data_X = np.array(data_X)
    data_Y = np.array(data_Y)
    return  data_X,data_Y



def Normalize(list):
    list = np.array(list)
    low, high = np.percentile(list, [0, 100])
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = (list[i]-low)/delta
    return  list,low,high

def FNoramlize(list,low,high):
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = list[i]*delta + low
    return list

def Normalize2(list,low,high):
    list = np.array(list)
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = (list[i]-low)/delta
    return  list


def Train_Model(train_X,train_Y):
    model = Sequential()
    model.add(LSTM(4, input_shape=(train_X.shape[1],train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_X, train_Y, epochs=1000, batch_size=1, verbose=2)
    # model.save(os.path.join("DATA","LSTMBLog" + ".h5"))
    return model

alldata = pd.read_csv("./international-airline-passengers.csv")
#只取数值列
alldata = alldata.iloc[:,1]
alldata = np.array(alldata,dtype='float64')
traindata = alldata[:int(len(alldata)*0.6)]
testdata = alldata[int(len(alldata)*0.6):]
print(traindata,testdata)
'''
#实验1
train_n,train_low,train_high = Normalize(traindata)
test_n,test_low,test_high = Normalize(testdata)
print(train_n,test_n)
#前一个值预测后一个值
train_X,train_Y = Create_dataset(train_n,look_back=1)
test_X,test_Y = Create_dataset(test_n,look_back=1)
#额外添加一个维度使train_X，test_X变为三维
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

model = Train_Model(train_X,train_Y)

train_predict = model.predict(train_X)
test_predict  = model.predict(test_X)

#反归一化
train_Y = FNoramlize(train_Y,train_low,train_high)
train_predict = FNoramlize(train_predict,train_low,train_high)
test_Y = FNoramlize(test_Y,test_low,test_high)
test_predict = FNoramlize(test_predict,test_low,test_high)

#进行绘图
plt.subplot(121)
plt.plot(train_Y)
plt.plot(train_predict)
plt.subplot(122)
plt.plot(test_Y)
plt.plot(test_predict)
plt.show()
'''

'''
#实验2
train_n,train_low,train_high = Normalize(traindata)
#更新了
test_n = Normalize2(testdata,train_low,train_high)
print(train_n,test_n)
#前一个值预测后一个值
train_X,train_Y = Create_dataset(train_n,look_back=1)
test_X,test_Y = Create_dataset(test_n,look_back=1)
#额外添加一个维度使train_X，test_X变为三维
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

model = Train_Model(train_X,train_Y)

train_predict = model.predict(train_X)
test_predict  = model.predict(test_X)

#反归一化
train_Y = FNoramlize(train_Y,train_low,train_high)
train_predict = FNoramlize(train_predict,train_low,train_high)
test_Y = FNoramlize(test_Y,train_low,train_high)
test_predict = FNoramlize(test_predict,train_low,train_high)

#进行绘图
plt.subplot(121)
plt.plot(train_Y)
plt.plot(train_predict)
plt.subplot(122)
plt.plot(test_Y)
plt.plot(test_predict)
plt.show()
'''
#实验3
alldata_n,all_low,all_high = Normalize(alldata)
#也可以直接对alldata进行截取
train_n = Normalize2(traindata,all_low,all_high)
test_n = Normalize2(testdata,all_low,all_high)
print(train_n,test_n)

#前一个值预测后一个值
train_X,train_Y = Create_dataset(train_n,look_back=1)
test_X,test_Y = Create_dataset(test_n,look_back=1)
#额外添加一个维度使train_X，test_X变为三维
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

model = Train_Model(train_X,train_Y)

train_predict = model.predict(train_X)
test_predict  = model.predict(test_X)

#反归一化
train_Y = FNoramlize(train_Y,all_low,all_high)
train_predict = FNoramlize(train_predict,all_low,all_high)
test_Y = FNoramlize(test_Y,all_low,all_high)
test_predict = FNoramlize(test_predict,all_low,all_high)

#进行绘图
plt.subplot(121)
plt.plot(train_Y)
plt.plot(train_predict)
plt.subplot(122)
plt.plot(test_Y)
plt.plot(test_predict)
plt.show()
