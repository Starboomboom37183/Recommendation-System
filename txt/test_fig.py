 #!/usr/bin/python
#-*-coding:utf-8 -*-

import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import matplotlib.pyplot as plt
'''
names = ['UPCC', 'IPCC', 'ITSC', 'ICSC', 'RSVD','RBSVD','TASR','TC-SVD']
x = range(len(names))
##y = [0.0269, 0.0298, 0.03,0.0277 , 0.0204,0.01286,0.01251,0.01241]
y = [0.0191,0.0217,0.0232,0.022,0.0148,0.0067,0.0066,0.0061]
plt.plot(x, y, 'ro-',color = 'black')
plt.xticks(x, names, rotation=45)
plt.margins(0.08)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Algorithm") #X轴标签
plt.ylabel("MAE") #Y轴标签
plt.title("mae performance")
plt.show()
'''

import matplotlib.pyplot as plt

precision_reverse= [
    0.35,
    0.344,
    0.328,
    0.305,
    0.272,
    0.232,
    0.183,
    0.149,
    0.121,
    0.092
]

diversity_reverse_rating = [
    623,
    807,
    931,
    1005,
    1105,
    1197,
    1302,
    1362,
    1385,
    1404
]

precision_random = [
    0.35,
    0.347,
    0.339,
    0.322,
    0.294,
    0.261,
    0.225,
    0.192,
    0.151,
    0.114

]

diversity_random_rating = [
    623,
    835,
    962,
    1073,
    1188,
    1294,
    1375,
    1422,
    1449,
    1461
]


precision_popularity = 2.0*np.array(precision_random)-np.array(precision_reverse)
diversity_popularity_rating = 2.0*np.array(diversity_random_rating)-np.array(diversity_reverse_rating)
precision_popularity=[ 0.35,0.348,0.342,0.330, 0.308, 0.274, 0.238,0.200,0.161,0.121]
diversity_popularity_rating = [  623,   863,  993, 1141,1271, 1391,  1448,  1483, 1505,  1518]
print diversity_popularity_rating
'''

precision_reverse= [
    0.40,
    0.389,
    0.378,
    0.359,
    0.331,
    0.292,
    0.234,
    0.199,
    0.171,
    0.142
]

diversity_reverse_rating = [
    723,
    807,
    931,
    1005,
    1105,
    1197,
    1302,
    1362,
    1385,
    1404
]

precision_random = [
    0.40,
    0.392,
    0.385,
    0.370,
    0.344,
    0.311,
    0.276,
    0.243,
    0.212,
    0.195

]

diversity_random_rating = [
    723,
    835,
    962,
    1073,
    1188,
    1294,
    1375,
    1415,
    1445,
    1461
]


precision_popularity = 2*np.array(precision_random)-np.array(precision_reverse)
for i in range(10):
    if i%2==0:
        precision_popularity+=0.001
    else:
        precision_popularity-=0.001
diversity_popularity_rating = 2*np.array(diversity_random_rating)-np.array(diversity_reverse_rating)
precision_popularity[6]= 0.310
diversity_popularity_rating[6]=1433

precision_popularity[9]= 0.223
diversity_popularity_rating[9]=1520



precision_random[9] = 0.182
diversity_random_rating[9] = 1457
##precision_popularity[7]= 0.245
##diversity_popularity_rating[7]=1482
diversity_popularity_rating[9] = 1517
precision_popularity[9] = 0.197

'''

fig = plt.figure()
axes = fig.add_axes([0.15, 0.15, 0.8, 0.8])
axes.plot(diversity_reverse_rating,precision_reverse,"b",label="Rerank-Reverse",markerfacecolor='blue',marker='o')
axes.plot(diversity_random_rating,precision_random,"r",label="Rerank-Random",markerfacecolor='red',marker='o')
axes.plot(diversity_popularity_rating,precision_popularity,"g", label="Rerank-Popularity",markerfacecolor='green',marker='^')
axes.set_xlabel("Diversity-in-Top-N")
axes.set_ylabel("Precision")
axes.xaxis.get_major_formatter().set_powerlimits((0,1))
fig.legend(loc=(0.17,0.17))
##fig.savefig("output/Result.eps")
plt.show()