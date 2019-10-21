#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 00:24:23 2019

@author: theone
"""

import LR as lr
import numpy as np
import math
import LDA
file_path1= "winequality-red.csv"
file_path2="breast-cancer-wisconsin.data"

def genDataWOHeader (file_path):
    data = np.genfromtxt(file_path, delimiter = ";" , skip_header = 1)
    return data
def genData (file_path):
    data = np.genfromtxt(file_path, delimiter = ",")
    return data

#categolize the last column of redwine data to 0 and 1
#@param data: dataset
def qualityToCategory (data):
    for i in range(data.shape[0]):
        if data[i,-1] >5:
            data[i,-1] = 1
        else:
            data[i,-1] = 0
            
#categorize the last column of cancer data to 0 and 1
#@param data: dataset
def classToCategory (data):
    for i in range(data.shape[0]):
        if data[i,-1] ==4:
            data[i,-1] = 1
        else:
            data[i,-1] = 0
    
#seperate test and train data 
#@param data: whole dataset
#@return testSet: testSet
#@return trainSet: trainSet
def seperateTestSet(data):
    dataSize= data.shape[0]
    testSize=math.floor(dataSize/10)
    testSet= data[:testSize, :]
    trainSet = data[testSize:, :]
    
    return testSet, trainSet

#remove the rows with 'nan' in the dataset
#@Param data: whole dataset
def preprocessData(data):
    
    data=data[~np.isnan(data).any(axis=1)]
    return data


            
# remove the dateset has 3 features outside 2* standard deviation
# @Param data: dataset
#@return: cleaned data
def removeOutLiersByND (data):
    mean = []
    SD = []
    for i in range(len(data[0]) - 1):
        mean.append(np.mean(data[:, i]))
        SD.append(np.std(data[:, i]))
    
    delet_arr = np.array([])
    for i in range(len(data)):
        count = 0
        for j in range(len(data[0]) - 1):
            if abs(data[i, j] - mean[j]) > 2*SD[j]:
                count += 1
        if count >= 3:
            delet_arr = np.append(delet_arr, i)
    np.unique(delet_arr)
    clean_data = np.delete(data, delet_arr, axis = 0)
    return clean_data

data = genDataWOHeader(file_path1)
qualityToCategory(data)
data2 = removeOutLiersByND (data)
print(len(data))
print(len(data2))
    