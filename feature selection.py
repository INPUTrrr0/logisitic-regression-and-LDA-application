#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:32:08 2019

@author: theone
"""

# -*- coding: utf-8 -*-
    
import logistic_regression as lr
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

#select subset of all features by choosing the best
#performing subset of all subsets given a model
#@param model: an instance of a model
#@Param data: whole dataset
#@return bestPerformingFeatures : the set of features that performs the best
def featureSelection (data, isLR):
    selectedFeatureNum = []
    selectedFeatureArray = -1 
    bestAccuracyAll = 0
    y_2d = np.array([data[:, -1]]).T
    #print(y_2d)
    for i in range(data.shape[1]-1):
        featureToAdd = -1
        bestAccuracy = 0
        column_2d = -1
        print("select feature{}".format(i))
        if i==0:
            for j in range(data.shape[1]-1):
                if (j in selectedFeatureNum ) == False:
                    column_2d = np.array([data[:, j]]).T
                    nums = selectedFeatureNum + [j]
                    
                    # ------5 should be changed --
                    #print(np.concatenate((column_2d,y_2d), axis = 1))
                    if isLR:
                        model=lr.LogisticRegression(0.001,500)
                        accuracy = LRKFoldValidation(model, np.concatenate((column_2d,y_2d), axis = 1),5)
                    else:
                        model=LDA.LDA()
                        accuracy = LDAKFoldValidation(model, np.concatenate((column_2d,y_2d), axis = 1),5)

                    print ("Using feature(s){} accuracy is{}".format(nums, accuracy))
                    if accuracy >= bestAccuracy :
                        bestAccuracy = accuracy
                        featureToAdd = j
            selectedFeatureArray = column_2d 
            bestAccuracyAll = bestAccuracy
            selectedFeatureNum.append(featureToAdd)
            continue
        else:   
            #try add feature from the rest of set
            for j in range(data.shape[1]-1):
                if (j in selectedFeatureNum ) == False:
                    column_2d = np.array([data[:, j]]).T
                    nums = selectedFeatureNum + [j]
                    
                    # ------5 should be changed ---
                    #print(np.concatenate((selectedFeatureArray, column_2d , y_2d), axis = 1))
                    if isLR:
                        model=lr.lr.LogisticRegression(0.001,500)
                        accuracy = LRKFoldValidation(model, np.concatenate((selectedFeatureArray, column_2d , y_2d), axis = 1),5)
                    else:
                        model=LDA.LDA
                        accuracy = LDAKFoldValidation(model, np.concatenate((selectedFeatureArray, column_2d , y_2d), axis = 1),5)
                    print ("Using feature(s){} accuracy is{}".format(nums, accuracy))
                    if accuracy >= bestAccuracy :
                        bestAccuracy = accuracy
                        featureToAdd = j
                        
        #additional feature cannot improve performance by 1%
        if bestAccuracyAll >= bestAccuracy:
            print ("maxima reached")
            break
        else:
            #add addtional feature
            bestAccuracyAll = bestAccuracy
            selectedFeatureNum.append(featureToAdd)
            selectedFeatureArray = np.concatenate((selectedFeatureArray,np.array([data[:, featureToAdd]]).T),axis =1)
    print("feature selection ended, best performing features are {}, the accuracy is {}".format(selectedFeatureNum, bestAccuracyAll))
    return selectedFeatureNum, selectedFeatureArray
        

#removeOutlier data to z-score by each feature except the last column
#@Param data: dataset 
#@return data: the normalized data set
def removeOutlier(data):
    for i in range(data.shape[1]-1):
        mean = np.mean(data[: , i])
        deviation = np.std(data[: , i])
        for j in range(data.shape[0]):
            data[i,j] = (data[i,j] - mean)/deviation
    return data     

#do k fold validation
#@param model: the instance of the model
#@param trainData: the train data
#@param k: the number of folds for validation
#@return accuracy: the accuracy of the model on the given dataset
def LDAKFoldValidation (model, trainData, k):
        size = math.floor(trainData.shape[0] / k)
        error = 0
        for i in range(k):
            if (i == k-1):
                validationSet = trainData[(i * size) : , :]
                trainSet = trainData[ : (i * size) , :]
            else:
                validationSet = trainData[(i * size): ((i + 1) * size), :]
                trainSet = np.concatenate((trainData[0 : i* size, :], trainData[(i + 1)* size : , :]),axis=0)
                
            x_train= trainSet[: , :-1]
            x_validation=validationSet[: , :-1]
            y_train= trainSet[:, -1]
            #normalize(x_train)
            #normalize(x_validation)
            #print(x_train.shape)
            p0, p1, u0, u1, covariance = model.fit(x_train, y_train)
            
            Y_predict = model.predict(x_validation,u0,u1,p0,p1,covariance)
            Y_true = validationSet[:, -1]
            count = 0
            for j in range(Y_true.shape[0]):
                if Y_predict[j] != Y_true[j]:
                    count +=1
            #print("count is{}".format(count ))
            error += count
            
        avg_error = error / k
        accuracy= 1- avg_error / (trainData.shape[0]/k)
        return accuracy

def normalize(data):
        m = data.shape[1]
        for i in range(m-1):
                data[:,i] = np.divide(data[:,i] - np.min(data[:,i]),np.max(data[:,i]) - np.min(data[:,i]))
                
def LRKFoldValidation(model,data, k) :
        '''
        size = math.floor(data.shape[1] / k)
        error = 0
        for i in range(k):
            if (i == k-1):
                validationSet = data[(i * size) : , :]
                trainSet = data[ : (i * size) , :]
            else:
                validationSet = data[(i * size): ((i + 1) * size), :]
                trainSet = np.concatenate((data[0 : i* size, :], data[(i + 1)* size : , :]),axis=0)
            x_train= trainSet[: , :-1]
            x_validation = validationSet [:, :-1]
            y_train= trainSet[:, -1]
            
            normalize(x_train)
            normalize(x_validation)
          #  for i in range(x_train.shape[1]):
           #     x_train[:,i] = np.divide(x_train[:,i] - np.min(x_train[:,i]),np.max(x_train[:,i]) - np.min(x_train[:,i]))
            #    x_validation[:,i] = np.divide(x_validation[:,i] - np.min(x_validation[:,i]),np.max(x_validation[:,i]) - np.min(x_validation[:,i]))
            w= model.fit(x_train,y_train)
            Y_predict = model.predict(x_validation,w)
            Y_true = validationSet[:, -1]
            count = 0
            for j in range(len(Y_true)):
                if Y_predict[j] != Y_true[j]:
                    count +=1
            print("count is{}".format(count ))
            error += count
            
        avg_error = error / k
        print("# of data is {}".format(data.shape[0]))
        error_percentage= avg_error / (data.shape[0]/k)
        print(error_percentage)
        
        return count/ data.shape[0] 
        '''
        size = int(len(data) / k)
        error = 0
        for i in range(k):
            if (i == k-1):
                validationSet = data[(i * size) : , :]
                trainSet = data[ : (i * size) , :]
            else:
                validationSet = data[(i * size): ((i + 1) * size), :]
                trainSet = np.concatenate((data[0 : i* size, :], data[(i + 1)* size : , :]),axis=0)
            x_train= trainSet[: , :-1]
            x_validation = validationSet [:, :-1]
            y_train= trainSet[:, -1]
            
          #  for i in range(x_train.shape[1]):
           #     x_train[:,i] = np.divide(x_train[:,i] - np.min(x_train[:,i]),np.max(x_train[:,i]) - np.min(x_train[:,i]))
            #    x_validation[:,i] = np.divide(x_validation[:,i] - np.min(x_validation[:,i]),np.max(x_validation[:,i]) - np.min(x_validation[:,i]))
            w= model.fit(x_train,y_train)
            Y_predict = model.predict(x_validation,w)
            Y_true = validationSet[:, -1]
            count = 0
            for j in range(len(Y_true)):
                if Y_predict[j] != Y_true[j]:
                    count += 1
            #print("count is{}".format(count ))
            error += count
            #print(count)
            
        avg_error = error / k
        error_percentage= avg_error / (data.shape[0]/k)
        
        return 1- error_percentage
    
def addSquareFeature(data, array):
    for i in array:
        colToAdd = np.power(data[: , i],3)
        data = np.insert(data, -1, colToAdd, axis = 1 )
    return data
        
    
def testLRWithWine(a, epochs):
    data = genDataWOHeader(file_path1)
    qualityToCategory(data)
    np.random.shuffle(data)
    #data1= removeOutLiersByND(data2)
    testSet, trainSet = seperateTestSet(data)
    #trainSet=np.insert(trainSet, trainSet.shape[1]-1,np.ones((trainSet.shape[0],1),dtype=float),axis=1)
    aModel= lr.LogisticRegression(a,epochs)
    return LRKFoldValidation(aModel,data, 5)
    
def testLRWithCancer(a, epochs):
    data = genData(file_path2)
    classToCategory(data)
    np.random.shuffle(data)
    preprocessData(data)

    #data1= removeOutLiersByND(data2)
    testSet, trainSet = seperateTestSet(data)
    #trainSet=np.insert(trainSet, trainSet.shape[1]-1,np.ones((trainSet.shape[0],1),dtype=float),axis=1)
    aModel= lr.LogisticRegression(a,epochs)
    return LRKFoldValidation(aModel,data, 5)
    
def testLDAWithWine():
    data = genDataWOHeader(file_path1)
    qualityToCategory(data)
    np.random.shuffle(data)
    #data1= removeOutLiersByND(data2)
    testSet, trainSet = seperateTestSet(data)
    aModel = LDA.LDA()
    return LDAKFoldValidation(aModel,trainSet, 5)

def testLDAWithCancer():
    data = genData(file_path2)
    classToCategory(data)
    preprocessData(data)
    np.random.shuffle(data)
    #data1= removeOutLiersByND(data2)
    testSet, trainSet = seperateTestSet(data)
    aModel = LDA.LDA()
    return LDAKFoldValidation(aModel,trainSet, 5)

#accuracy = testLRWithWine(0.01, 500)
#print("Accuracy is{}".format(accuracy))
    

#initialize wine data
data1 = genDataWOHeader(file_path1)
qualityToCategory(data1)
np.random.shuffle(data1)
normalize(data1)
wineTestSet, wineTrainSet = seperateTestSet(data1)

#initialize cancer data
data2 = genData(file_path2)
data2 = preprocessData(data2)
classToCategory(data2)
normalize(data2)
np.random.shuffle(data2)
cancerTestSet, cancerTrainSet = seperateTestSet(data2)

#data1= removeOutLiersByND(data2)
#shuffle the data
#data1= removeOutLiersByND(data2)
#testSet, trainSet = seperateTestSet(data2)
#aModel = LDA.LDA()

#for i in range(5):
 #   np.random.shuffle(data1)
  #  featureSelection(LRModel,data1,True)
#LDAKFoldValidation(aModel,trainSet, 5)
print("LR-Wine-Feature",featureSelection (wineTrainSet, True))
  

'''
a = 0
b = 0
c = 0
d = 0
for i in range(3):
    np.random.shuffle(data1)
    np.random.shuffle(data2)
    a +=LRKFoldValidation(LRModel, data1, 5)
    b +=LDAKFoldValidation(LDAModel,data1 , 5)
    c +=LRKFoldValidation(LRModel, data2, 5)
    d +=LDAKFoldValidation(LDAModel, data2, 5)

print(a/3)
print(b/3)
print(c/3)
print(d/3)

normalize(data1)
normalize(data2)

a2 = 0
b2 = 0
c2 = 0
d2 = 0
for i in range(3):
    np.random.shuffle(data1)
    np.random.shuffle(data2)
    a2 +=LRKFoldValidation(LRModel, data1, 5)
    b2 +=LDAKFoldValidation(LDAModel,data1 , 5)
    c2 +=LRKFoldValidation(LRModel, data2, 5)
    d2 +=LDAKFoldValidation(LDAModel, data2, 5)

print(a2/3)
print(b2/3)
print(c2/3)
print(d2/3)

data1r= removeOutLiersByND(data1)
data2r= removeOutLiersByND(data2)

a3 = 0
b3 = 0
c3 = 0
d3 = 0
for i in range(3):
    np.random.shuffle(data1r)
    np.random.shuffle(data2r)
    a3 +=LRKFoldValidation(LRModel, data1r, 5)
    b3 +=LDAKFoldValidation(LDAModel,data1r , 5)
    c3 +=LRKFoldValidation(LRModel, data2r, 5)
    d3 +=LDAKFoldValidation(LDAModel, data2r, 5)

print(a3/3)
print(b3/3)
print(c3/3)
print(d3/3)


data3 = addSquareFeature( data1, [10,1,9,6])  
normalize(data1)
normalize(data3)
a1 = 0;
a2 = 0;
a4 = 0;
a3 = 0;
for i in range(5):
    np.random.shuffle(data1)
    np.random.shuffle(data3)
    a1+=LRKFoldValidation(LRModel, data1, 5)
    a2+=LRKFoldValidation(LRModel, data3, 5)
    np.random.shuffle(data1)
    np.random.shuffle(data3)
    a3+=LDAKFoldValidation(LDAModel, data1, 5)
    a4+=LDAKFoldValidation(LDAModel, data3, 5)

print(a1/5)
print(a2/5)
print(a3/5)
print(a4/5)
'''


#print(LRKFoldValidation(LRModel, data1, 5))
#print(LDAKFoldValidation(LDAModel, data2, 5))
print(LRKFoldValidation(LRModel, cancerTrainSet, 5))
print(LDAKFoldValidation(LDAModel, cancerTrainSet, 5))

# learning rate: 0.0001 - 1, Iteration: 50 - 100000
bestLearn = 0
bestIte = 0
learn = [0.001, 0.01, 0.1, 1]
ite = [100, 500, 1000, 5000]
max_acc = 0
