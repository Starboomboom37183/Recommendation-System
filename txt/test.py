# !/usr/bin/python
# _*_ coding:utf-8 _*_
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
import sklearn
def split_dataset(file):
    input = open(file)
    dataset = []
    for line in input:
        dataset.append(line)
    shuffle(dataset)
    train_x, test_x = train_test_split(dataset, test_size=0.1, random_state=0)
    output1 = open('train.txt','w')
    output2 = open('test.txt','w')
    for i in range(len(train_x)):
        output1.write(train_x[i])
    for i in range(len(test_x)):
        output2.write(test_x[i])

def load_train():
    input1 = open('User.txt')
    input2 = open('webservice.txt')
    ws = {}
    us = {}
    for line in input2:
        s = line.strip('\r\n').split(' ')
        ##print s[0],s[1]
        ws[s[0]] = int(s[1])
    for line in input1:
        s = line.strip('\r\n').split(' ')
        us[s[0]] = int(s[1])
    matrix = np.zeros((510, 4813))
    test =   np.zeros((510,4813))
    input3 = open('train.txt')
    input4 = open('test.txt')
    for line in input3:
        s = line.strip('\r\n').split(' ')
        i = us[s[0]]
        j = ws[s[1]]
        if (matrix[i][j] != 0):
            matrix[i][j] += float(s[3])
        ##print s[3]
        else:
            matrix[i][j] = float(s[3])

    for line in input4:
        s = line.strip('\r\n').split(' ')
        i = us[s[0]]
        j = ws[s[1]]
        if (test[i][j] != 0):
            test[i][j] += float(s[3])
        else:
            test[i][j] = float(s[3])
    return matrix,test


##print np.sum(train!=0)
def cal_sparsity(matrix):
    total = np.sum(matrix!=0)+np.sum(matrix==0)
    return (1-float(np.sum(matrix!=0))/total)

def cal_cosine_similarity(matrix):
    ##return sklearn.metrics.pairwise.cosine_similarity(matrix, dense_output=True)
    return pairwise_distances(matrix, metric='cosine')

def cal_pearson_similarity(matrix):
    b = matrix.mean(axis = 1)
    matrix_ = matrix - b[:,np.newaxis]
    sim = pairwise_distances(matrix_, metric='cosine')
    return sim

def IPCC(ratings,similarity):
    pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return np.nan_to_num(pred)
def UPCC(ratings,similarity):
    mean_user_rating = ratings.mean(axis=1)
    ##print mean_user_rating.shape
    # You use np.newaxis so that mean_user_rating has same format as ratings
    ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    return pred

def cal_presion_diversity(matrix,test,pred, N):
    recall = 0.0
    precision = 0.0
    hit =0
    x = set()
    a = np.argsort(pred,axis=1)
    Top_N = []
    for i in range(matrix.shape[0]):
        Top = []
        for j in range(a.shape[1]):
            if(matrix[i][a[i][j]]>0):
                j+=1;
            else:
                Top.append(a[i][j])
            if(len(Top)==N):
                break;
        Top_N.append(Top)
    c = np.sum(test!=0,axis =1)
    for i in range(len(Top_N)):
        hit = 0
        for j in range(N):
            x.add(Top_N[i][j])
            if test[i][Top_N[i][j]]>0:

                hit+=1
        if c[i]!=0:
            recall+=(hit*1.0)/c[i]
        precision+=(hit*1.0)/N


    recall = (recall*1.0)/np.sum(c!=0)
    precision = (precision*1.0)/510
    print recall
    print precision
    print len(x)
'''
split_dataset('final_file_1.txt')
train,test = load_train()
sim = cal_cosine_similarity(train.T)
sim2 = cal_cosine_similarity(train)
pred = IPCC(train,sim)
pred2 = UPCC(train,sim2)
cal_presion_diversity(train,test,pred,5)
cal_presion_diversity(train,test,pred2,5)
'''


class Service(object):
    def __init__(self, ws, rating, popularity):
        self.__ws = ws
        self.__rating = rating
        self.__popularity = popularity
    # 取得age属性
    def getWsName(self):
        return self.__ws
    def getRating(self):
        return self.__rating
    def getPopularity(self):
        return self.__popularity
    def printWS(self):
        return self.__ws,self.__rating,self.__popularity


a = [8,4,2,6,7,3,3,9,0,5]
popularity =[20,10,30,45,33,11,56,61,21,9]
list = []
list.append(a)
list.append(popularity)
print np.unique(list)
'''
for i in range(len(a)):
    list.append(Service(i,a[i],popularity[i]))


##for Tr in range(10):
Tr = 3
list = sorted(list, reverse=1,key=lambda )
k = 0
for i in range(len(list)):
    if list[i].getRating()<Tr:
            k = i
            break

a = list[:k]
b = list[k:]
a = sorted(a, reverse=0,key=lambda Service:Service.getPopularity())
list = a+b
for i in range(len(list)):
    print list[i].printWS()
result = []
for i in range(5):
    result.append(list[i].getWsName())
print result
'''

