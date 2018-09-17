# !/usr/bin/python
# _*_ coding:utf-8 _*_
import numpy as np
import xlrd
import  sklearn.metrics
def get_composition_matrix():
    data = xlrd.open_workbook('Composition_mashup1.xlsx')
    table = data.sheets()[0]
    nrows = table.nrows
    c_m = {}
    for i in range(1,nrows):
        str = table.row_values(i)[0][8:]
        str = str.strip().lower()
        str = str.replace(' ','-')
        ##print str
        list1 = table.row_values(i)[1].split(',')
        for i in range(len(list1)):
            list1[i] = list1[i].strip().lower()
            list1[i] = list1[i].replace(' ','-')
        ##print list1
        c_m[str] = list1

    input = open('webservice.txt')
    ws = {}
    mashup = []
    API = {}
    ##0-2712 api
    for line in input:
        s = line.strip('\r\n').split(' ')
        ##print s[0],s[1]
        ws[s[0]] = int(s[1])
        if(int(s[1])>2712):
            mashup.append(s[0])
        else:
            API[s[0]] = int(s[1])

    count = 0
    ##2712-3444  mashup找不到
    for m in mashup:
        if c_m.has_key(m):
            count+=1
            ##print m
        else:
            k = len(API)
            API[m] = k
    ###3444 api找不到
    for m in mashup:
        if c_m.has_key(m):
            for api in c_m[m]:
                if API.has_key(api):
                    continue
                else:
                    ##print m,api
                    k = len(API)
                    API[api] = k

    c_matrix = np.zeros((4813,3774))
    for w,c in ws.items():
        if c<=2712:
            c_matrix[c][API[w]] = 1
        else:
            ##print w,c_m[w]
            if c_m.has_key(w):
                for api in c_m[w]:
                    c_matrix[c][API[api]] = 1
            else:
                ##print API[w]
                c_matrix[c][API[w]] = 1

    #####   #####
    sim1 = sklearn.metrics.pairwise.cosine_similarity(c_matrix, dense_output=True)
    matrix = c_matrix[:,0:2713].T
    sim2 = sklearn.metrics.pairwise.cosine_similarity(matrix, dense_output=True)
    ##print np.sum(sim2!=0)
    for i in range(2713):
        for j in range(2713):
            sim1[i,j] = sim2[i,j]

    return sim1
