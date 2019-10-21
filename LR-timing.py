   
import LR as lr
import numpy as np
import math
import LDA
import time
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

def normalize(data):
        m = data.shape[1]
        for i in range(m-1):
                data[:,i] = np.divide(data[:,i] - np.min(data[:,i]),np.max(data[:,i]) - np.min(data[:,i]))

start_time = time.time()
def LRKFoldValidation(model,data, k) :
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
            
        avg_error = error / k
        error_percentage= avg_error / (data.shape[0]/k)
        
        return 1- error_percentage
    
end_time = time.time()
#initialize wine data
dataraw = genDataWOHeader(file_path1)
qualityToCategory(dataraw)
np.random.shuffle(dataraw)
data2 = removeOutLiersByND(dataraw)
normalize(data2)
np.random.shuffle(data2)
wineTestSet, wineTrainSet = seperateTestSet(data2)

    
Model=lr.LogisticRegression(0.001,500)
np.random.shuffle(dataraw)
testSet, trainSet = seperateTestSet(dataraw)

print("LR-wine-clean", LRKFoldValidation(Model,wineTrainSet,5))


#elapsed_time = timeit.timeit(code_to_test, number=100)/100
#print(elapsed_time)
print("--- %s seconds ---" % (end_time - start_time))