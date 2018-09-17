# !/usr/bin/python
# _*_ coding:utf-8 _*_
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import sys
import numpy as np
import matplotlib
import scipy
import matplotlib.pyplot as plt
import lda
import lda.datasets
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import  sklearn.metrics
from get_mashup import MSSSQL

def get_LDA_doc():
    ms = MSSSQL(host="172.28.4.193", user="sa", pwd="wy9756784750", db="pweb")
    resList = ms.ExeQuery("select wsname,summary,categoryid,tags,Description from dbo.ws")
    ws = {}
    for (wsname,summary,categoryid,tags,Description) in resList:
        s1 = wsname.strip().lower().replace(' ','-')
        ll = summary+' '+categoryid+' '+tags+' '+Description
        ll = ll.replace('\n',' ')
        ws[s1] = ll
    mp = {}
    resList = ms.ExeQuery("select wsname,summary,categoryid,tags,Description from dbo.mp")
    for (wsname,summary,categoryid,tags,Description) in resList:
        s1 = wsname.strip().lower().replace(' ','-')
        ll = summary + ' ' + categoryid + ' ' + tags + ' ' + Description
        ll = ll.replace('\n', ' ')
        ws[s1] = ll

    input = open('webservice.txt')
    output = open('LDA_train.txt','w')
    cc = 0
    for line in input:
        s = line.strip('\r\n').split(' ')
        cc+=1
        if ws.has_key(s[0]):
            output.write(ws[s[0]]+'\r\n')
        elif mp.has_key(s[0]):
            output.write(mp[s[0]]+'\r\n')
        else:
            output.write(s[0]+'\r\n')
    print cc
    input.close()
    output.close()
##get_LDA_doc()
def get_topic_sim(k):
    corpus = []
    input = open('LDA_result.txt')
    for line in input:
        corpus.append(line.strip('\r\n'))
    print len(corpus)

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频

    vectorizer = CountVectorizer()
    print 'vectorizer',vectorizer

    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()
    weight = X.toarray()
    # LDA算法
    print 'LDA:'


    model = lda.LDA(n_topics=k, n_iter=300, random_state=1)
    model.fit(np.asarray(weight))  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works

    # 文档-主题（Document-Topic）分布
    doc_topic = model.doc_topic_
    ##print("type(doc_topic): {}".format(type(doc_topic)))
    ##print("shape: {}".format(doc_topic.shape))
    ##print doc_topic
    sim = sklearn.metrics.pairwise.cosine_similarity(doc_topic, dense_output=True)
    ##print sim
    return sim
'''
label = []
for n in range(100):
    topic_most_pr = doc_topic[n].argmax()
    label.append(topic_most_pr)
    print("doc: {} topic: {}".format(n, topic_most_pr))


'''
def get_k_nearest(sim,k):
    Nt = []
    for i in range(sim.shape[0]):
        temp = np.argpartition(-sim[i], k+1)[:k+1]
        temp = np.delete(temp,np.where(temp==i),axis=0)
        Nt.append(temp)

    return Nt

##sim = get_topic_sim()
'''
def get_sim_top_k_name(sim,k):
    Nt = get_k_nearest(sim,5)
    input = open('webservice.txt')
    s_name = {}
    for line in input:
        s = line.strip('\r\n').split(' ')
        s_name[int(s[1])] = s[0]
    print Nt
    print s_name[587]
    print Nt[587]
    for i in Nt[587]:
        print s_name[i]
    return Nt
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
    input3 = open('final_file_1.txt')
    for line in input3:
        s = line.strip('\r\n').split(' ')
        i = us[s[0]]
        j = ws[s[1]]
        matrix[i][j] = 1
    return matrix
sim = get_topic_sim()
Nt = get_sim_top_k_name(sim,20)
matrix = load_train('')
count = 0
for i in range(510):
    if matrix[i][587]==1:
        count+=1
print count
c = []
for k in Nt[587]:
    print k
    temp = 0
    for i in range(510):
        if  matrix[i][k]==1:
            temp+=1
    c.append(temp)
print c
'''

##get_topic_sim()