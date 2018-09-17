# !/usr/bin/python
# _*_ coding:utf-8 _*_
import xlrd
import operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def draw_bar(labels,quants,xname,yname,title):
    width = 0.4
    ind = np.linspace(0.5,9.5,10)
    # make a square figure
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    # Bar Plot
    ax.bar(ind-width/2,quants,width,color='green')
    # Set the ticks on x-axis
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    # labels
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    # title
    ax.set_title(title, bbox={'facecolor':'0.8', 'pad':5})
    plt.grid(False)

    for a, b in zip(ind, quants):
        ##print a,b
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
    plt.show()
    plt.savefig("bar.png")
    plt.close()


data = xlrd.open_workbook('Composition_mashup1.xlsx')

table = data.sheets()[0]

nrows = table.nrows

dict = {}
dict1 = {}
api =  {}
sum = 0
A = []
B = []
for i in range(1,nrows):
    str = table.row_values(i)[0][8:]
    list1 = table.row_values(i)[1].split(',')

    for apiName in list1:
        apiName = apiName.strip()
        if(apiName=='Google Maps'):
            A.append(str)
        if(apiName=='Facebook'):
            B.append(str)
        if(api.has_key(apiName)):
            api[apiName]+=1
        else:
            api[apiName]=1

    count =  len(list1)
    sum+=count
    if(dict.has_key(str)):
        print str
    dict[str] = list1
    if(dict1.has_key(count)):
        dict1[count] +=1
    else:
        dict1[count] = 1


##print dict
print len(dict)

print nrows

print dict1
##统计平均每个mashup调用的web service api的个数
print round(sum,2)/nrows
##print api
print len(api)
apiList = sorted(api.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

##统计api被调用的情况
##for i in range(10):
    ##print apiList[i][0]
    ##print api[apiList[i][0]]

print apiList


data = []
name = []
for i in range(10):
    data.append(apiList[i][1])
    name.append(apiList[i][0])
print name
draw_bar(name,data,'API Name','No.','Top 10 Most Popular Api')

sum  = 0
for key,value in dict1.items():
    if key>=6:
        sum+=dict1[key]
print sum

for i in range(len(A)):
    if(A[i] in B):
        print A[i]
        print dict[A[i]]
print B


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 6)
y = x * x
plt.plot(x, y, marker='o')
for xy in zip(x, y):
    plt.annotate("Tr = %s" % y, xy=xy, xytext=(-20, 10), textcoords='offset points')
plt.show()