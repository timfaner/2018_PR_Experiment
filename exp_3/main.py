# -*- coding:utf-8 -*-

from math import log
import os
os.chdir('./exp_3')

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    dataSet.sort(key = lambda x:x[axis]) #sort by  attribute
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def splitBinDataSet(dataSet, axis, value):
    binedRetDataSet = [[],[]]
    RawRetDataSet = [[],[]]
    for data in dataSet:
        i = 0 if data[axis] <= value else 1
        a = data[:axis];b = data[:axis]
        a.append(i);b.append(data[axis])
        a.extend(data[axis+1:]);b.extend(data[axis+1:])
        binedRetDataSet[i].append(a)
        RawRetDataSet.append(b)
    return binedRetDataSet,RawRetDataSet
            


def calcAttributeShannonEnt(dataSet,contiunLabel):
    '输入数据集，以及每个属性是否离散的标签'
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    attribute_information_list =[]
    iv_list = []
    split_point_list = [-1 for i in range(numFeatures)]
    for i in range(numFeatures):
        
        if contiunLabel[i]:  
            dataSet.sort(key = lambda x:x[i])
            featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
            uniqueVals = set(featList) 
            newEntropy = 0.0
            newIV = 0.0
            for value in uniqueVals:
                E = 0.0
                IV = 0.0
                subBinDataSet_list = splitBinDataSet(dataSet,i,value)[0]
                for subBinDataSet in subBinDataSet_list:
                    prob = len(subBinDataSet)/float(len(dataSet))
                    e = calcShannonEnt(subBinDataSet)
                    if e == 0 or prob == 1:
                        continue
                    E += prob * e
                    IV -= prob/log(prob,2)
                if E > newEntropy:newEntropy = E;newIV=IV;split_point_list[i] = value
                if E == 1.5743313880649628:
                    print(1)

        else:      
            featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
            uniqueVals = set(featList)       #get a set of unique values
            newEntropy = 0.0
            newIV = 0.0
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))
                newEntropy += prob * calcShannonEnt(subDataSet)
                newIV -= prob/log(prob,2)     
        attribute_information_list.append(newEntropy)
        iv_list.append(newIV)
            
    return attribute_information_list,iv_list ,split_point_list                     #returns list

def getImformativeFeature(atr_l,iv_l,dataSet):
    gain_list = []
    gain_ratio_list = []
    baseE = calcShannonEnt(dataSet)
    sumE = 0
    for i in range(len(atr_l)):
        gain_list.append(baseE-atr_l[i])
        gain_ratio_list.append((baseE-atr_l[i])/iv_l[i])
        sumE += atr_l[i]
    avrE = sumE/len(atr_l)


    index_list = []
    for i in range(len(atr_l)):
        if atr_l[i] <= avrE:
            index_list.append(i)
    #print(index_list)
    av_gr_dict = {index:gain_ratio_list[index] for index in index_list}

    max_gr = 0;attr_index = -1
    for index,value in av_gr_dict.items():
        if value > max_gr:
            max_gr = value;attr_index = index

    return attr_index

def createTree(dataSet,labels,contiunLabel):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    if len(dataSet[0]) == 1: 
        return majorityCnt(classList)
    atr_l,iv_l,splt_l = calcAttributeShannonEnt(dataSet,contiunLabel)
    bestFeat = getImformativeFeature(atr_l,iv_l,dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    if splt_l[bestFeat] == -1:
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            del(contiunLabel[bestFeatLabel])
            subLabels = labels.copy()     
            myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels,contiunLabel)
    else:
        subLabels = labels.copy()  
        subDataList = splitBinDataSet(dataSet, bestFeat, splt_l[bestFeat])[1]
        for subData in subDataList:
            myTree[bestFeatLabel][value] = createTree(subData,subLabels,contiunLabel)

    return myTree  

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def main():
    
    data_set = []
    with open('./bezdekIris.data.txt','r') as f:
        m = f.readlines()
        for line in m[:-1]:
            a = line.strip().split(',')

            
            sample = list(map(float,a[:-1]))
            sample.append(a[-1])
            data_set.append(sample)

    continueLabel = [1,1,1,1]
    print(createTree(data_set,['a','b','c','d'],continueLabel))


    

    
if __name__ == '__main__':
    main()