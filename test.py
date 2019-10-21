# -*- coding: utf-8 -*-
import numpy as np
import math
from main import *
import logistic_regression as lr
import LDA
#initialize wine data


def genRW():
    data1 = genDataWOHeader("winequality-red.csv")
    qualityToCategory(data1)
    np.random.shuffle(data1)
    return data1

#initialize cancer data
def genCancer():
    data2 = genData("breast-cancer-wisconsin.data")
    data2 = preprocessData(data2)
    classToCategory(data2)
    np.random.shuffle(data2)
    return data2




def genRWNormalized():
    rwNormalized = genDataWOHeader("winequality-red.csv")
    qualityToCategory(rwNormalized)
    np.random.shuffle(rwNormalized)
    data= normalize(rwNormalized)
    return data
    
def genCancerNormalized():
    #initialize cancer data
    cancerNormalized = genData("breast-cancer-wisconsin.data")
    cancerNormalized = preprocessData(cancerNormalized)
    classToCategory(cancerNormalized)
    np.random.shuffle(cancerNormalized)
    data =  normalize(cancerNormalized)
    return data
#data1= removeOutLiersByND(data2)

def genRWRemovedOL():
    return removeOutLiersByND(genRW())
def genCancerRemovedOL ():
    return removeOutLiersByND(genRW())

def genRWClear ():
    return removeOutLiersByND(genRWNormalized())

def genCancerClear ():
    return removeOutLiersByND(genCancerNormalized())


def testSelectedFeatures1():
    print("start testAdditionlSquaredFeatures()" )
    LRModel = lr.LogisticRegression(0.001, 500)
    LDAModel =LDA.LDA()
    data1 = genRWNormalized()
    data2 = np.append(data1[: ,[10,1,9,6]], np.array([data1[:, -1]]).T, axis =1)
    data3 = addSquareFeature(data1, [10,1,9,6])
    a1 = 0
    b1 = 0
    a2 = 0
    b2 = 0
    a3 = 0
    b3 = 0
    for i in range(3):
        np.random.shuffle(data1)
        np.random.shuffle(data2)
        np.random.shuffle(data3)
        a1 += LRKFoldValidation(LRModel,data1, 5)
        b1 += LDAKFoldValidation(LDAModel,data2, 5)
        a2 += LRKFoldValidation(LRModel,data2, 5)
        b2 += LDAKFoldValidation(LDAModel,data2, 5)
        a3 += LRKFoldValidation(LRModel,data3, 5)
        b3 += LDAKFoldValidation(LDAModel,data3, 5)
    print("Accuracy for lr in rw is {}".format(a1/3))
    print("Accuracy for LDA in rw is {}".format(b1/3))
    print("Accuracy for lr in rw is {}".format(a2/3))
    print("Accuracy for LDA in rw is {}".format(b2/3))
    print("Accuracy for lr in rw is {}".format(a3/3))
    print("Accuracy for LDA in rw is {}".format(b3/3))    
    
def testSelectedFeatures2():
    LRModel = lr.LogisticRegression(0.001, 500)
    data1 = genRWNormalized()
    square = np.copy(data1)
    for i in range(len(data1[0]) - 1):
        colToAdd = np.power(data1[: , i], 2)
        square = np.insert(square, -1, colToAdd, axis = 1 )
        square = np.insert(square, -1, np.multiply(data1[:, 0], data1[:, 8]), axis = 1)
        square = np.insert(square, -1, np.multiply(data1[:, 0], data1[:, 7]), axis = 1)
        square = np.insert(square, -1, np.multiply(data1[:, 0], data1[:, 2]), axis = 1)
        square = np.insert(square, -1, np.multiply(data1[:, 5], data1[:, 6]), axis = 1)
        LRModel = lr.LogisticRegression(0.001, 100)
        featureSelection(LRModel, square, 3)
def testDataPreprocess():
    rwData = genRW()
    cancerData =genCancer()
    rwNormalized = genRWNormalized()
    cancerNormalized =genCancerNormalized()
    rwRemovedOL = genRWRemovedOL()
    cancerRemovedOL = genCancerRemovedOL ()
    rwClear = genRWClear()
    cancerClear = genCancerClear()
    LRModel = lr.LogisticRegression(0.001, 500)
    LDAModel =LDA.LDA()
    
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(3):
        np.random.shuffle(rwData)
        np.random.shuffle(cancerData)
        a +=LRKFoldValidation(LRModel, rwData, 5)
        b +=LDAKFoldValidation(LDAModel,rwData , 5)
        c +=LRKFoldValidation(LRModel, cancerData, 5)
        d +=LDAKFoldValidation(LDAModel, cancerData, 5)
    
    print(a/3)
    print(b/3)
    print(c/3)
    print(d/3)
    
    a2 = 0
    b2 = 0
    c2 = 0
    d2 = 0
    for i in range(3):
        np.random.shuffle(rwNormalized)
        np.random.shuffle(cancerNormalized)
        a2 +=LRKFoldValidation(LRModel, rwNormalized, 5)
        b2 +=LDAKFoldValidation(LDAModel,rwNormalized , 5)
        c2 +=LRKFoldValidation(LRModel, cancerNormalized, 5)
        d2 +=LDAKFoldValidation(LDAModel, cancerNormalized, 5)
    print(a2/3)
    print(b2/3)
    print(c2/3)
    print(d2/3)
    
    a3 = 0
    b3 = 0
    c3 = 0
    d3 = 0
    for i in range(3):
        np.random.shuffle(rwClear)
        np.random.shuffle(cancerClear)
        a3 +=LRKFoldValidation(LRModel, rwClear, 5)
        b3 +=LDAKFoldValidation(LDAModel,rwClear , 5)
        c3 +=LRKFoldValidation(LRModel, cancerClear, 5)
        d3 +=LDAKFoldValidation(LDAModel, cancerClear, 5)
    print(a3/3)
    print(b3/3)
    print(c3/3)
    print(d3/3)
    
    a4 = 0
    b4 = 0
    c4 = 0
    d4 = 0
    for i in range(3):
        np.random.shuffle(rwRemovedOL)
        np.random.shuffle(cancerRemovedOL)
        a4 +=LRKFoldValidation(LRModel, rwRemovedOL, 5)
        b4 +=LDAKFoldValidation(LDAModel,rwRemovedOL , 5)
        c4 +=LRKFoldValidation(LRModel, cancerRemovedOL, 5)
        d4 +=LDAKFoldValidation(LDAModel, cancerRemovedOL, 5)
    print(a4/3)
    print(b4/3)
    print(c4/3)
    print(d4/3)


#print(LRKFoldValidation(LRModel, data1, 5))
#print(LDAKFoldValidation(LDAModel, data2, 5))
#print(LRKFoldValidation(LRModel, cancerTrainSet, 5))
#print(LDAKFoldValidation(LDAModel, cancerTrainSet, 5))

def testAlphaAndEpochs():
    rwClear = genRWClear()
    # learning rate: 0.0001 - 1, Iteration: 50 - 100000
    bestLearn = 0
    bestIte = 0
    learn = [0.001, 0.01, 0.1, 1]
    ite = [100, 500, 1000, 5000]
    max_acc = 0
    
    for i in learn:
        for j in ite:
            LRModel = lr.LogisticRegression(i, j)
            ave = 0.0
            for k in range(3):
                ac = LRKFoldValidation(LRModel, rwClear, 5)
                print("per k fold:", ac)
                ave += ac
            ave = ave / 3.0
            print("ave:", ave)
            if ave > max_acc:
                max_acc = ave
                bestLearn = i
                bestIte = j
            print(ave," ", i," ", j)
    print(bestLearn)
    print(bestIte)
    print(max_acc)
    

LRModel = lr.LogisticRegression(0.001, 500)
LDAModel =LDA.LDA()
rwNormalized = genRWNormalized()
cancerNormalized = genCancerNormalized()
rwNormalized = genRWNormalized()
print(LRKFoldValidation(LRModel, cancerNormalized, 5))
print(LDAKFoldValidation(LDAModel, cancerNormalized, 5))
print(LRKFoldValidation(LRModel, rwNormalized, 5))
print(LDAKFoldValidation(LDAModel, rwNormalized, 5))
