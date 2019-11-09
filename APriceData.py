import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

InputRows = 30  # 取前面多少行作为训练数据输入

RawData = pd.read_csv('1.csv', encoding='gbk')

print(RawData.head())

UsefulData = RawData[['成交额', '开盘价', '最低价', '最高价', '收盘价']]

UsefulData = np.array(UsefulData)
# InputRows行数据输入，列数为原始数据列数-InputRows
x_Train = np.zeros(((np.size(UsefulData, 0) - InputRows - 1), InputRows * 5))
y_Train = np.zeros((np.size(UsefulData, 0) - InputRows) - 1)
# 从UseData 取前面inputrows行，作为Trainx的一行

print(x_Train.shape)
print(y_Train.shape)

def GetTrainData(irows, inputData, outputx, outputy):
    rowsize = np.size(inputData, 1)  # 有用数据一行几个数据
    columnsize = np.size(inputData, 0)  # 测试数据多少行
    print("rowsize=%d columnsize=%d" %(rowsize,columnsize))
    for datacolumn in range(irows, columnsize-1):  # 列
        for row in range(datacolumn - irows, datacolumn):  # 取前InputRows行数据给x_Train当前行
            for column in range(rowsize):  # 每行的数据个数
                if(datacolumn > 4110):
                    print("datacolumn=%d irows=%d row=%d rowsize=%d " %(datacolumn,irows,row,rowsize))
                outputx[datacolumn - irows][(row + irows - datacolumn) * rowsize + column] = inputData[row][column]
        outputy[datacolumn - irows] = (inputData[row + 1][rowsize - 1] - inputData[row][rowsize - 1]) / inputData[row + 1][rowsize - 1]

GetTrainData(InputRows, UsefulData, x_Train, y_Train)


print(x_Train)
print(y_Train)

a = 0
for i in range(np.size(y_Train,0)):
    if(y_Train[i] > 0):
        a = a +1
        print(y_Train[i]*100)

print(a)















