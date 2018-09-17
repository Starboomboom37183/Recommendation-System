# !/usr/bin/python
# _*_ coding:utf-8 _*_
import numpy as np
from random import  shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
import sklearn.metrics.pairwise
import sklearn
from LDA_study import get_topic_sim
from composition_matrix import get_composition_matrix
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

def load_train(file):
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
def get_ws_user_no():
    input1 = open('User.txt')
    input2 = open('webservice.txt')
    ws = {}
    us = {}
    for line in input2:
        s = line.strip('\r\n').split(' ')
        ws[s[0]] = int(s[1])
    for line in input1:
        s = line.strip('\r\n').split(' ')
        us[s[0]] = int(s[1])
    return ws,us
def load_data_tuple(file):
    ws,us = get_ws_user_no()
    input = open(file)
    train_dataset = []
    ##max_ut = 0   2997
    ##max_st = 0   2970
    for line in input:
        s = line.strip('\r\n').split(' ')
        list = []
        list.append(int(us[s[0]]))
        list.append(int(ws[s[1]]))
        list.append(float(s[3]))
        list.append(int(s[4]))
        list.append(int(s[5]))
        ##print list
        train_dataset.append(tuple(list))
    return train_dataset



def cal_sparsity(matrix):
    total = np.sum(matrix!=0)+np.sum(matrix==0)
    return (1-float(np.sum(matrix!=0))/total)


##split_dataset('final_file_1.txt')
def cal_cosine_similarity(matrix):
    return sklearn.metrics.pairwise.cosine_similarity(matrix, dense_output=True)
from scipy import stats

def cal_pearson_similarity(matrix):
    b = matrix.mean(axis = 1)
    matrix_ = matrix - b[:,np.newaxis]
    sim = sklearn.metrics.pairwise.cosine_similarity(matrix_, dense_output=True)

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
    return np.nan_to_num(pred)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
def cal_rmse(prediction,ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
def cal_mae(prediction,ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return mean_absolute_error(prediction,ground_truth);

'''
user_sim =  cal_cosine_similarity(matrix)
item_sim = cal_cosine_similarity(matrix.T)
user_p_sim = cal_pearson_similarity(matrix)
item_p_sim = cal_pearson_similarity(matrix.T)
##print sim.shape
##

pred = UPCC(matrix,user_sim)
pred1 = UPCC(matrix,user_p_sim)
pred2 = IPCC(matrix,item_sim)
pred3 = IPCC(matrix,item_p_sim)
print cal_rmse(pred,test)
print cal_rmse(pred1,test)

'''


def Rsvd(matrix):
    n_factors = 20  # number of factors
    alpha = 0.5  # learning rate
    lamda = 0.003
    n_epochs = 100
    p = np.random.normal(0, .005, (matrix.shape[0], n_factors))
    q = np.random.normal(0, .005, (matrix.shape[1], n_factors))
    index = np.argwhere(matrix!=0)
    ##print len(index)
    for _ in range(n_epochs):
        for (u,i) in index:
            err = matrix[u][i] - np.dot(p[u], q[i])
            # Update vectors p_u and q_i
            p[u] += alpha *( err * q[i] - lamda*p[u])
            q[i] += alpha * (err * p[u] - lamda*q[i])
        alpha = 0.9*alpha
    return np.dot(p,q.T)


def rsvd(mat, feature=20, steps = 300, gama = 1, lamda = 0.001):
    # feature是潜在因子的数量，mat为评分矩阵
    slowRate = 0.9
    preRmse = 0.0000000000001
    nowRmse = 0.0

    user_feature = np.random.normal(0, .005, (mat.shape[0], feature))
    item_feature = np.random.normal(0, .005, (mat.shape[1], feature))
    print mat.shape[0]
    for step in range(steps):
        rmse = 0.0
        n = 0
        for u in range(510):
            for i in range(mat.shape[1]):
                if mat[u,i]>0:
                    # 这边是判断是否为空，也可以改为是否为0:if mat[u,i]>0:
                    pui = float(np.dot(user_feature[u, :], item_feature[i, :].T))
                    eui = mat[u, i] - pui
                    rmse += pow(eui, 2)
                    n += 1
                        # Rsvd的更新迭代公式
                    user_feature[u] += gama * (eui * item_feature[i] - lamda * user_feature[u])
                    item_feature[i] += gama * (eui * user_feature[u] - lamda * item_feature[i])
                        # n次迭代平均误差程度
        nowRmse = sqrt(rmse * 1.0 / n)
        ##print 'step: %d      Rmse: %s' % ((step + 1), nowRmse)
        if (nowRmse > preRmse):
            pass
        else:
            break
        # 降低迭代的步长
        gama *= slowRate
        step += 1
    return np.dot(user_feature,item_feature.T)


##rb_svd is a svd model where biases are involved
def rb_svd(mat, feature=20, steps = 50, gama = 0.01, lamda = 0.001):

    # feature是潜在因子的数量，mat为评分矩阵
    preRmse = 0.0000000000001
    nowRmse = 0.0

    user_feature = np.random.normal(0, .0005, (mat.shape[0], feature))
    item_feature = np.random.normal(0, .0005, (mat.shape[1], feature))
    bu = np.random.normal(0, .0005, (mat.shape[0], 1))
    ##print bu.shape
    bi = np.random.normal(0, .0005, (mat.shape[1], 1))
    print mat.shape[0]
    slowRate = 0.9
    for step in range(steps):
        rmse = 0.0
        n = 0
        for u in range(510):
            for i in range(mat.shape[1]):
                if mat[u,i]>0:
                    # 这边是判断是否为空，也可以改为是否为0:if mat[u,i]>0:
                    pui = bu[u]+bi[i]+float(np.dot(user_feature[u, :], item_feature[i, :].T))
                    eui = mat[u, i] - pui
                    rmse += pow(eui, 2)
                    n += 1
                        # Rsvd的更新迭代公式
                    user_feature[u] += gama * (eui * item_feature[i] - lamda * user_feature[u])
                    item_feature[i] += gama * (eui * user_feature[u] - lamda * item_feature[i])
                    bu[u] +=gama*(eui- lamda*bu[u])
                    bi[i] +=gama*(eui-lamda*bi[i])

                        # n次迭代平均误差程度
        nowRmse = sqrt(rmse * 1.0 / n)
        print 'step: %d      Rmse: %s' % ((step + 1), nowRmse)
        if (nowRmse > preRmse):
            pass
        else:
            break
        # 降低迭代的步长
        gama *= slowRate
        step += 1
    pred_m = np.dot(user_feature,item_feature.T)
    for u in range(mat.shape[0]):
        for i in range(mat.shape[1]):
            pred_m[u,i] = pred_m[u,i]+bu[u]+bi[i]
    return pred_m
##论文方法 TASR论文中的方法
def TASR(mat,test,steps=70,feature=20,alpha=0.9,lamda=0.001):
    train_set = load_data_tuple('train.txt')
    test_set = load_data_tuple('test.txt')
    slowRate = 0.99
    nowRmse = 0.0
    user_feature = np.random.normal(0, .0005, (mat.shape[0], feature))
    item_feature = np.random.normal(0, .0005, (mat.shape[1], feature))
    bu = np.zeros((mat.shape[0],1))
    bi = np.zeros((mat.shape[1],1))
    Pu = np.random.normal(0, .005, (mat.shape[0], feature))
    Qut = np.random.normal(0, .005, (2998, feature))
    Rs = np.random.normal(0, .005, (mat.shape[1], feature))
    Sst = np.random.normal(0, .005, (2991, feature))
    Vl = np.random.normal(0, .005, (mat.shape[0], feature))
    Vm = np.random.normal(0, .005, (mat.shape[1], feature))
    Vnut = np.random.normal(0, .005, (2998, feature))

    for step in range(steps):
        rmse = 0.0
        n = 0
        for (u,s,r,ut,st) in train_set:
            temp = 0
            n+=1
            for k in range(feature):
                temp+=Vl[u,k]*Vm[s,k]*Vnut[ut,k]
            eui = mat[u,s]-(bu[u]+bi[s]+np.dot(user_feature[u],item_feature[s])+np.dot(Pu[u],Qut[ut])+np.dot(Rs[s],Sst[st])+temp)
            rmse += pow(eui, 2)
            bu[u] += alpha*(eui-lamda*bu[u])
            bi[s] += alpha*(eui-lamda*bi[s])
            ##print eui
            user_feature[u] += alpha * (eui * item_feature[s] - lamda * user_feature[u])
            item_feature[s] += alpha * (eui * user_feature[u] - lamda * item_feature[s])
            Pu[u] += alpha*(eui*Qut[ut] - lamda*Pu[u])
            Qut[ut]+=alpha*(eui*Pu[u] - lamda*Qut[ut])
            Rs[s] += alpha*(eui*Sst[st] - lamda*Rs[s])
            Sst[st]+=alpha*(eui*Rs[s] - lamda*Sst[st])
            for k in range(feature):
                Vl[u,k]+=alpha*(eui*np.dot(Vm[s],Vnut[ut]) - lamda*Vl[u,k])
                Vm[s, k] += alpha * (eui * np.dot(Vl[u], Vnut[ut]) - lamda * Vm[s, k])
                Vnut[ut, k] += alpha * (eui * np.dot(Vl[u], Vm[s]) - lamda * Vnut[ut, k])

        nowRmse = sqrt(rmse * 1.0 / n)
        print 'step: %d      Rmse: %s' % ((step + 1), nowRmse)
        step+=1
        alpha = alpha*slowRate
    print 'calculate test rmse:'
    n = 0
    rmse = 0.0
    mae = 0.0
    for u,s,r,ut,st in test_set:
        temp = 0.0
        for k in range(feature):
            temp += Vl[u, k] * Vm[s, k] * Vnut[ut, k]
        eui = test[u,s]  - (bu[u] + bi[s] + np.dot(user_feature[u], item_feature[s]) + np.dot(Pu[u], Qut[ut]) + np.dot(Rs[s],Sst[st]) + temp)
        rmse += pow(eui, 2)
        mae +=abs(eui)
        n+=1
    final_rmse = sqrt(rmse * 1.0 / n)
    print 'final rmse is ',final_rmse
    print 'final mae is ',(mae*1.0/n)
    return final_rmse,(mae*1.0/n)

def get_k_nearest(sim,k):
    Nt = []
    for i in range(sim.shape[0]):
        temp = np.argpartition(-sim[i], k+1)[:k+1]
        temp = np.delete(temp,np.where(temp==i),axis=0)
        Nt.append(temp)
    return Nt

def TimeSVD(mat,test_mat,topic_sim,composition_sim,steps=120,feature=20,alpha=0.9,lamda=0.001,k1 = 20):
    t_Nt = get_k_nearest(topic_sim,k1)
    c_Nt =  get_k_nearest(composition_sim,k1)
    train_set = load_data_tuple('train.txt')
    test_set = load_data_tuple('test.txt')
    slowRate = 0.99
    nowRmse = 0.0
    user_feature = np.random.normal(0, .005, (mat.shape[0], feature))
    item_feature = np.random.normal(0, .005, (mat.shape[1], feature))
    bu = np.random.normal(0, .005, (mat.shape[0], 1))
    bi = np.random.normal(0, .005, (mat.shape[1], 1))
    Pu = np.random.normal(0, .005, (mat.shape[0], feature))
    Qut = np.random.normal(0, .005, (2998, feature))
    Rs = np.random.normal(0, .005, (mat.shape[1], feature))
    Sst = np.random.normal(0, .005, (2991, feature))
    Vl = np.random.normal(0, .005, (mat.shape[0], feature))
    Vm = np.random.normal(0, .005, (mat.shape[1], feature))
    Vnut = np.random.normal(0, .005, (2998, feature))
    x1 = 0.2
    x2 = 0.2
    for step in range(steps):
        rmse = 0.0
        n = 0
        for (u,s,r,ut,st) in train_set:
            temp = 0.0
            n+=1
            for k in range(feature):
                temp+=Vl[u,k]*Vm[s,k]*Vnut[ut,k]
            total_sim = 0.0
            temp1 = np.zeros((feature,))
            for i1 in t_Nt[s]:
                total_sim += topic_sim[s, i1]
                temp1 += topic_sim[s, i1] * item_feature[i1]
            if total_sim == 0:
                temp1 = 0
            else:
                temp1 /= total_sim
            ##组合相似计算
            temp2 = np.zeros((feature,))
            total_sim = 0.0
            for i1 in c_Nt[s]:
                total_sim += composition_sim[s, i1]
                temp2 += composition_sim[s, i1] * item_feature[i1]
            if total_sim == 0:
                temp2 = 0
            else:
                temp2 /= total_sim
            final_temp = np.zeros((feature,))
            final_temp = (1 - x1 - x2) * item_feature[s] + x1 * temp1 + x2 * temp2

            eui = r-(bu[u]+bi[s]+float(np.dot(user_feature[u],final_temp.T))+np.dot(Pu[u],Qut[ut])+np.dot(Rs[s],Sst[st])+temp)
            rmse += pow(eui, 2)
            bu[u] += alpha*(eui-lamda*bu[u])
            bi[s] += alpha*(eui-lamda*bi[s])
            ##print eui
            user_feature[u] += alpha * (eui * final_temp - lamda * user_feature[u])
            item_feature[s] += alpha * (eui * user_feature[u] - lamda * item_feature[s])
            Pu[u] += alpha*(eui*Qut[ut] - lamda*Pu[u])
            Qut[ut]+=alpha*(eui*Pu[u] - lamda*Qut[ut])
            Rs[s] += alpha*(eui*Sst[st] - lamda*Rs[s])
            Sst[st]+=alpha*(eui*Rs[s] - lamda*Sst[st])
            for k in range(feature):
                Vl[u,k]+=alpha*(eui*np.dot(Vm[s],Vnut[ut]) - lamda*Vl[u,k])
                Vm[s, k] += alpha * (eui * np.dot(Vl[u], Vnut[ut]) - lamda * Vm[s, k])
                Vnut[ut, k] += alpha * (eui * np.dot(Vl[u], Vm[s]) - lamda * Vnut[ut, k])

        nowRmse = sqrt(rmse * 1.0 / n)
        print 'step: %d      Rmse: %s' % ((step + 1), nowRmse)
        step+=1
        alpha = alpha*slowRate
        print 'calculate test rmse:'
        n = 0
        rmse = 0.0
        for u,s,r,ut,st in test_set:
            temp = 0.0
            for k1 in range(feature):
                temp += Vl[u, k1] * Vm[s, k1] * Vnut[ut, k1]
            total_sim = 0.0
            temp1 = np.zeros((feature,))
            for i1 in t_Nt[s]:
                total_sim += topic_sim[s, i1]
                temp1 += topic_sim[s, i1] * item_feature[i1]
            if total_sim == 0:
                temp1 = 0
            else:
                temp1 /= total_sim
            ##组合相似计算
            temp2 = np.zeros((feature,))
            total_sim = 0.0
            for i1 in c_Nt[s]:
                total_sim += composition_sim[s, i1]
                temp2 += composition_sim[s, i1] * item_feature[i1]
            if total_sim == 0:
                temp2 = 0
            else:
                temp2 /= total_sim
            final_temp = np.zeros((feature,))
            final_temp = (1 - x1 - x2) * item_feature[s] + x1 * temp1 + x2 * temp2
            eui = r  - (bu[u] + bi[s] + float(np.dot(user_feature[u], final_temp.T)) + np.dot(Pu[u], Qut[ut]) + np.dot(Rs[s],Sst[st]) + temp)
            rmse += pow(eui, 2)
            n+=1
        final_rmse = sqrt(rmse * 1.0 / n)
        print 'test rmse is ',final_rmse

    ##return final_rmse

def rb_svd_new(mat,test_mat, topic_sim,composition_sim,k,feature=20, steps = 800, gama = 0.9, lamda = 0.001):###train 0.00817, test 0.0147

    # feature是潜在因子的数量，mat为评分矩阵
    t_Nt = get_k_nearest(topic_sim,k)
    c_Nt =  get_k_nearest(composition_sim,k)
    slowRate = 0.99
    preRmse = 0.0000000000001
    nowRmse = 0.0
    ##x1 = 0.2
    ##x2 = 0.1
    user_feature = np.random.normal(0, .005, (mat.shape[0], feature))
    item_feature = np.random.normal(0, .005, (mat.shape[1], feature))
    Yt =  np.random.normal(0, .005, (mat.shape[1], feature))
    Yc =  np.random.normal(0, .005, (mat.shape[1], feature))

    bu = np.random.normal(0, .005, (mat.shape[0], 1))
    ##print bu.shape
    bi = np.random.normal(0, .005, (mat.shape[1], 1))
    print mat.shape[0]
    for step in range(steps):
        rmse = 0.0
        n = 0
        for u in range(510):
            for i in range(mat.shape[1]):
                if mat[u,i]>0:
                    # 这边是判断是否为空，也可以改为是否为0:if mat[u,i]>0:
                    total = 0
                    temp1 = np.zeros((feature,))
                    for i1 in t_Nt[i]:
                        total+=1
                        temp1+=Yt[i1]
                    if total == 0:
                        temp1 = 0
                    else:
                        temp1/=sqrt(total)
                    ##组合相似计算
                    temp2 = np.zeros((feature,))
                    total1 = 0
                    for i1 in c_Nt[i]:
                        total1+=1
                        temp2+=Yc[i1]
                    if total1 == 0:
                        temp2 = 0
                    else:
                        temp2/=sqrt(total1)
                    final_temp = np.zeros((feature,))
                    final_temp = item_feature[i]+temp1+temp2
                    pui = bu[u]+bi[i]+float(np.dot(user_feature[u], final_temp.T))
                    eui = mat[u, i] - pui
                    rmse += pow(eui, 2)
                    n += 1
                        # Rsvd的更新迭代公式
                    user_feature[u] += gama * (eui * item_feature[i] - lamda * user_feature[u])
                    item_feature[i] += gama * (eui *user_feature[u] - lamda * item_feature[i])
                    if total!=0:
                        for i1 in t_Nt[i]:
                            Yt[i1] += gama * (eui*user_feature[u]/sqrt(total) - lamda * Yt[i1])
                    if total1!=0:
                        for i1 in c_Nt[i]:
                            Yc[i1] += gama * (eui * user_feature[u]/sqrt(total1) - lamda * Yc[i1])
                    bu[u] +=gama*(eui- lamda*bu[u])
                    bi[i] +=gama*(eui-lamda*bi[i])

        nowRmse = sqrt(rmse * 1.0 / n)
        print 'step: %d      Rmse: %s' % ((step + 1), nowRmse)
        # 降低迭代的步长
        gama *= slowRate
        step += 1

        print 'final test rmse: '
        n = 0
        rmse = 0.0
        mae = 0.0
        for u in range(510):
            for i in range(test_mat.shape[1]):
                if test_mat[u, i] > 0:
                    total = 0
                    temp1 = np.zeros((feature,))
                    for i1 in t_Nt[i]:
                        total += 1
                        temp1 += Yt[i1]
                    if total == 0:
                        temp1 = 0
                    else:
                        temp1 /= sqrt(total)
                    ##组合相似计算
                    temp2 = np.zeros((feature,))
                    total1 = 0
                    for i1 in c_Nt[i]:
                        total1 += 1
                        temp2 += Yc[i1]
                    if total1 == 0:
                        temp2 = 0
                    else:
                        temp2 /= sqrt(total1)
                    final_temp = np.zeros((feature,))
                    final_temp = item_feature[i] + temp1 + temp2
                    pui = bu[u] + bi[i] + float(np.dot(user_feature[u], final_temp))
                    eui = test_mat[u, i] - pui
                    rmse += pow(eui, 2)
                    mae+=abs(eui)
                    n += 1
        final_rmse =  sqrt(rmse * 1.0 / n)
        print 'final test rmse is ',final_rmse
        print 'final test mae is ',(mae*1.0/n)

def MyFunc(mat,test_mat, topic_sim,composition_sim,k,feature=30, steps = 250, gama = 0.9, lamda = 0.001):###train 0.00817, test 0.0147

    # feature是潜在因子的数量，mat为评分矩阵
    t_Nt = get_k_nearest(topic_sim,k)
    c_Nt =  get_k_nearest(composition_sim,k)
    slowRate = 0.99
    preRmse = 0.0000000000001
    nowRmse = 0.0
    x1 = 0.2
    x2 = 0.1
    user_feature = np.random.normal(0, .005, (mat.shape[0], feature))
    item_feature = np.random.normal(0, .005, (mat.shape[1], feature))
    bu = np.random.normal(0, .005, (mat.shape[0], 1))
    ##print bu.shape
    bi = np.random.normal(0, .005, (mat.shape[1], 1))
    print mat.shape[0]
    for step in range(steps):
        rmse = 0.0
        n = 0
        for u in range(510):
            for i in range(mat.shape[1]):
                if mat[u,i]>0:
                    # 这边是判断是否为空，也可以改为是否为0:if mat[u,i]>0:
                    total_sim = 0.0
                    temp1 = np.zeros((feature,))
                    for i1 in t_Nt[i]:
                        total_sim+=topic_sim[i,i1]
                        temp1+=topic_sim[i,i1]*item_feature[i1]
                    if total_sim == 0:
                        temp1 = 0
                    else:
                        temp1/=total_sim
                    ##组合相似计算
                    temp2 = np.zeros((feature,))
                    total_sim = 0.0
                    for i1 in c_Nt[i]:
                        total_sim+=composition_sim[i,i1]
                        temp2+=composition_sim[i,i1]*item_feature[i1]
                    if total_sim == 0:
                        temp2 = 0
                    else:
                        temp2/=total_sim
                    final_temp = np.zeros((feature,))
                    final_temp = item_feature[i]+x1*temp1+x2*temp2
                    pui = bu[u]+bi[i]+float(np.dot(user_feature[u], final_temp.T))
                    eui = mat[u, i] - pui
                    rmse += pow(eui, 2)
                    n += 1
                        # Rsvd的更新迭代公式
                    user_feature[u] += gama * (eui * final_temp - lamda * user_feature[u])
                    item_feature[i] += gama * (eui * user_feature[u] - lamda * item_feature[i])
                    bu[u] +=gama*(eui- lamda*bu[u])
                    bi[i] +=gama*(eui-lamda*bi[i])

        nowRmse = sqrt(rmse * 1.0 / n)
        print 'step: %d      Rmse: %s' % ((step + 1), nowRmse)
        # 降低迭代的步长
        gama *= slowRate
        step += 1

    print 'final test rmse: '
    n = 0
    rmse = 0.0
    mae = 0.0
    for u in range(510):
        for i in range(test_mat.shape[1]):
            if test_mat[u, i] > 0:
                total_sim = 0.0
                temp1 = np.zeros((feature,))
                for i1 in t_Nt[i]:
                    total_sim += topic_sim[i, i1]
                    temp1 += topic_sim[i, i1] * item_feature[i1]
                if total_sim == 0:
                    temp1 = 0
                else:
                    temp1 /= total_sim
                ##组合相似计算
                temp2 = np.zeros((feature,))
                total_sim = 0.0
                for i1 in c_Nt[i]:
                    total_sim += composition_sim[i, i1]
                    temp2 += composition_sim[i, i1] * item_feature[i1]
                if total_sim == 0:
                    temp2 = 0
                else:
                    temp2 /= total_sim
                final_temp = np.zeros((feature,))
                final_temp =  item_feature[i] + x1 * temp1 + x2 * temp2
                pui = bu[u] + bi[i] + float(np.dot(user_feature[u], final_temp.T))
                eui = test_mat[u, i] - pui
                rmse += pow(eui, 2)
                mae+=abs(eui)
                n += 1
    final_rmse =  sqrt(rmse * 1.0 / n)
    final_mae = mae*1.0/n
    print 'final test rmse is ',final_rmse
    print 'final test mae is ', final_mae
    '计算矩阵'
    final_item = np.zeros((mat.shape[1], feature))
    for i in range(mat.shape[1]):
        total_sim = 0.0
        temp1 = np.zeros((feature,))
        for i1 in t_Nt[i]:
            total_sim += topic_sim[i, i1]
            temp1 += topic_sim[i, i1] * item_feature[i1]
        if total_sim == 0:
            temp1 = 0
        else:
            temp1 /= total_sim
        ##组合相似计算
        temp2 = np.zeros((feature,))
        total_sim = 0.0
        for i1 in c_Nt[i]:
            total_sim += composition_sim[i, i1]
            temp2 += composition_sim[i, i1] * item_feature[i1]
        if total_sim == 0:
            temp2 = 0
        else:
            temp2 /= total_sim
        final_item[i] = item_feature[i] + x1 * temp1 + x2 * temp2

    pred_m = np.dot(user_feature, final_item.T)
    for u in range(mat.shape[0]):
        for i in range(mat.shape[1]):
            pred_m[u, i] = pred_m[u, i] + bu[u] + bi[i]
    return pred_m

    ##return final_rmse,final_mae
def SVD_average():
    ##split_dataset('final_file_1.txt')
    output = open('rb_svd_result1.txt','w')
    total_mae = 0.0
    total_rmse = 0.0
    for i in range(10):
        pred = rb_svd(matrix)
        mae = cal_mae(pred,test)
        rmse = cal_rmse(pred,test)
        output.write("第"+str(i)+"次: ")
        output.write("mae "+str(mae)+"\r\n")
        output.write("rmse "+str(rmse)+"\r\n")
        print "rmse ",rmse
        print "mae ",mae
        total_mae+=mae
        total_rmse+=rmse
    output.write("average mae "+str(total_mae*0.1)+"\r\n")
    output.write("average rmse "+str(total_rmse*(0.1))+"\r\n")
##split_dataset('final_file_1.txt')
matrix,test = load_train('')
##SVD_average()
##pred = rb_svd(matrix)    ##rmse 0.0138
##print cal_mae(pred,test)
##print cal_rmse(pred,test)
##topic_sim = get_topic_sim()
##pred = IPCC(matrix,topic_sim)
##print cal_rmse(pred,test)   ##rmse 0.031
##print cal_sparsity(matrix)
##load_data_tuple('')
##TASR(matrix,test)


##pred = rsvd(matrix,20,300,1,0.001)   ##rmse 0.0204
##pred = Rsvd(matrix)
##print cal_rmse(pred,test) ##0.0143
##print cal_rmse(pred,test)
##composition
##composition_sim = get_composition_matrix()
##pred = IPCC(matrix,composition_sim)
##print cal_rmse(pred,test)
##composition_sim = get_composition_matrix()
##print composition_sim.shape
##topic_sim = get_topic_sim()
##MyFunc(matrix,test,topic_sim,composition_sim)
##
## svd  k = 5 ,f =20 ,alpha = 0.9 lamda = 0.0001,   rmse = 0.0128
##rb_svd_new_2(matrix,test,topic_sim,composition_sim,5)

def SVD_new_average():
    output = open('influence_feature.txt','w')

    total_mae = 0.0
    total_rmse = 0.0
    topic_num = [10,20,30,40,50,60,70,80,90,100]

    for i in topic_num:
        topic_sim = get_topic_sim(i)
        rmse,mae = rb_svd_new_2(matrix,test,topic_sim,composition_sim,10,i)
        output.write("topic number is"+str(i)+":");
        output.write("mae"+str(mae)+"\r\n")
        output.write("rmse"+str(rmse)+"\r\n")
        total_mae+=mae
        total_rmse+=rmse
    ##output.write("average mae"+str(total_mae*0.1)+"\r\n")
    ##output.write("average rmse"+str(total_rmse*(0.1))+"\r\n")

def TASR_average():
    output = open('TASR.txt','w')
    total_mae = 0.0
    total_rmse = 0.0
    for i in range(5):
        rmse = 0.0
        mae = 0.0
        rmse,mae = TASR(matrix,test)
        output.write("第"+str(i)+"次: ")
        output.write("mae "+str(mae)+"\r\n")
        output.write("rmse "+str(rmse)+"\r\n")
        total_mae+=mae
        total_rmse+=rmse
    output.write("average mae "+str(total_mae*0.2)+"\r\n")
    output.write("average rmse "+str(total_rmse*(0.2))+"\r\n")
##SVD_average()
##SVD_new_average()
##rb_svd_new_2(matrix,test,topic_sim,composition_sim,10)
##rb_svd_new(matrix, test, topic_sim, composition_sim, 5)
##pred = rb_svd(matrix)    ##rmse 0.0138
##print cal_mae(pred,test)
##print cal_rmse(pred,test)
##user_sim =  cal_cosine_similarity(matrix)
##pred = UPCC(matrix,user_sim)
##print cal_mae(pred,test)
##print cal_rmse(pred,test)
'''
user_p_sim = cal_cosine_similarity(matrix)
item_p_sim = cal_pearson_similarity(matrix.T)

pred = UPCC(matrix,user_p_sim)
pred1 = IPCC(matrix,item_p_sim)
pred2 = IPCC(matrix,topic_sim)
pred3 = IPCC(matrix,composition_sim)
pred4 = Rsvd(matrix)
print cal_mae(pred,test)
print cal_mae(pred1,test)
print cal_mae(pred2,test)
print cal_mae(pred3,test)
print cal_mae(pred4,test)
'''
def cal_popularity(c):
    return np.sum(c!=0,axis =0)

def longtail_popularity():
    c = matrix+test
    t = cal_popularity(c)
    '''
    temp = np.argpartition(-t, 100)[:100]
    input = open('webservice.txt')
    ws = {}
    sum = 0
    total_sum = np.sum(t)
    print total_sum

    for line in input:
        s = line.strip('').split(' ')
        ##print s[0],s[1]
        ws[int(s[1])] = s[0]

    for i in temp:
        ##print ws[i],t[i]
        sum+=t[i]
    print sum
    print total_sum
    print (sum*1.0/total_sum)
'''

def get_bu_du(matrix,test):
    c = matrix + test
    t = cal_popularity(c)
    b_all = []
    d_all = []
    for i in range(matrix.shape[0]):
        temp_bu = []
        temp_du = []
        for j in range(matrix.shape[1]):
            if matrix[i][j]>0:
                temp_bu.append(matrix[i][j])
                temp_du.append(t[j])
        b_all.append(np.mean(temp_bu))
        d_all.append(np.std(temp_du))
    print b_all
    print np.sort(d_all)
    return b_all,d_all

def getTr(theta,b_all,d_all,T_max):
    TR =[]
    for i in range(len(b_all)):
        t = 0
        t = b_all[i]+(T_max-b_all[i])*theta/d_all[i]
        TR.append(t)
    return TR



def  get_Max_min(matrix):
    print np.max(matrix)
    print np.sum(matrix)*1.0/np.sum(matrix!=0)


##longtail_popularity()
##get_bu_du(matrix,test)

##get_Max_min(matrix)
##user_p_sim = cal_cosine_similarity(matrix)
##pred = UPCC(matrix,user_p_sim)
##item_p_sim = cal_pearson_similarity(matrix.T)
##pred = IPCC(matrix,item_p_sim)

##pred = rb_svd(matrix)

def rerank_popularity(list,tr,N):

    list = sorted(list, reverse=1, key=lambda Service: Service.getRating())
    k = 0
    for i in range(len(list)):
        if list[i].getRating() < tr:
            k = i
            break
    a = list[:k]
    b = list[k:]
    a = sorted(a, reverse=0, key=lambda Service: Service.getPopularity())
    list = a + b
    result = []
    for i in range(5):
        result.append(list[i].getWsName())
    return result

def rerank_reverse(list,popularity,tr,N):
    list = sorted(list, reverse=1, key=lambda Service: Service.getRating())
    k = 0
    for i in range(len(list)):
        if list[i].getRating() < tr:
            k = i
            break
    a = list[:k]
    b = list[k:]
    a = sorted(a, reverse=0, key=lambda Service: Service.getRating())
    list = a + b
    result = []
    for i in range(5):
        result.append(list[i].getWsName())
    return result

def rerank_random(list,popularity,tr,N):

    list = sorted(list, reverse=1, key=lambda Service: Service.getRating())
    k = 0
    for i in range(len(list)):
        if list[i].getRating() < tr:
            k = i
            break
    a = shuffle(list[:k])##随机
    b = list[k:]
    list = a + b
    result = []
    for i in range(5):
        result.append(list[i].getWsName())
    return result
##topic_sim = get_topic_sim(10)
##pred = rb_svd_new_2(matrix,test,topic_sim,composition_sim,10)
##pred = Rsvd(matrix)
def cal_presion_diversity(matrix,test,pred, N):
    recall = 0.0
    precision = 0.0
    hit =0
    x = set()
    a = np.argsort(-pred,axis=1)
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
            if test[i][Top_N[i][j]]>0:

                hit+=1
        if c[i]!=0:
            recall+=(hit*1.0)/c[i]
        precision+=(hit*1.0)/N


    recall = (recall*1.0)/np.sum(c!=0)
    precision = (precision*1.0)/510
    print recall
    print precision
    print len(np.unique(Top_N))
    ##return recall


##cal_Recall(matrix,test,pred,N=10)
def get_Rerank_result(matrix,test,pred,tr,N):
    popularity = cal_popularity(matrix)
    Top_N = []
    for i in range(pred.shape[0]):
        list = []
        for j in range(pred.shape[1]):
            if matrix[i][j]>0:
                continue
            else:
                list.append(Service(j,pred[i][j],popularity[j]))
        top = rerank_popularity(list,tr,N)
        Top_N.append(top)
    precision = 0
    for i in range(len(Top_N)):
        hit = 0
        for j in range(N):
            if test[i][Top_N[i][j]] > 0:
                hit += 1
        precision += (hit * 1.0) / N
    precision = (precision *1.0) / 510
    diversity = len(np.unique(Top_N))
    print precision
    print diversity


user_p_sim = cal_cosine_similarity(matrix)
item_sim  = cal_cosine_similarity(matrix.T)
pred = UPCC(matrix,user_p_sim)
##pred2 = IPCC(matrix,item_sim)
##cal_presion_diversity(matrix,test,pred,5)
##cal_presion_diversity(matrix,test,pred2,N=5)

##get_Rerank_result(matrix,test,pred,0,5)




