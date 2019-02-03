import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import  pandas as pd
import  os
from keras.models import Sequential, load_model





#多维归一化
def NormalizeMult(data):
    data = np.array(data)
    normalize = np.arange(2*data.shape[1],dtype='float64')

    normalize = normalize.reshape(data.shape[1],2)
    for i in range(0,data.shape[1]):
        #第i列
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        normalize[i,0] = listlow
        normalize[i,1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    #np.save("./normalize.npy",normalize)
    return  data,normalize


#多维反归一化
def FNormalizeMult(data,normalize):
    data = np.array(data)
    for i in  range(0,data.shape[1]):
        listlow =  normalize[i,0]
        listhigh = normalize[i,1]
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  data[j,i]*delta + listlow

    return data

#使用训练数据的归一化
def NormalizeMultUseData(data,normalize):
    data = np.array(data)
    for i in range(0, data.shape[1]):
        #第i列
        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta

    return  data

#单维归一化
def NormalizeSingle(list):
    list = np.array(list)
    normalizel = np.zeros(2)
    low,high = np.percentile(list, [0, 100])
    normalizel[0] = low
    normalizel[1] = high
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = (list[i] - low) / delta
    # np.save("./normalizel.npy",normalizel)
    return  list,normalizel

#单维归一化
def FNoramlizeSingle(list,normalizel):
    list = np.array(list)
    low = normalizel[0]
    high = normalizel[1]
    delta = high - low
    if delta != 0:
        for i in  range(0,len(list)):
            list[i] = list[i]*delta + low
    return  list
