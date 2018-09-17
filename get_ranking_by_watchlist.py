# !/usr/bin/python
# _*_ coding:utf-8 _*_
import pymssql
import xlrd
from get_mashup import MSSSQL
from datetime import *

class item:
    def __init__(self):
        self.lasttime = 0
        self.count = 0


def transfer_time(s):
    temp = s.split(' ')
    str = ''
    str+=temp[3]
    if(temp[2]=='Jan'):
        str+='-1'
    elif temp[2]=='Feb':
        str+='-2'
    elif temp[2]=='Mar':
        str+='-3'
    elif temp[2]=='Apr':
        str+='-4'
    elif temp[2]=='May':
        str+='-5'
    elif temp[2]=='Jun':
        str+='-6'
    elif temp[2]=='Jul':
        str+='-7'
    elif temp[2]=='Aug':
        str+='-8'
    elif temp[2]=='Sep':
        str+='-9'
    elif temp[2]=='Oct':
        str+='-10'
    elif temp[2]=='Nov':
        str+='-11'
    elif temp[2]=='Dec':
        str+='-12'
    str+= '-'
    str+=temp[1]
    str+=' '
    str+=temp[4]
    d1= datetime.strptime(str, '%Y-%m-%d %H:%M:%S')
    d2 = datetime.strptime('2005-09-03 17:00:02','%Y-%m-%d %H:%M:%S')
    d = d1-d2
    return d.days*24*3600+d.seconds

ms = MSSSQL(host="172.28.4.193", user="sa", pwd="wy9756784750", db="pweb")
resList = ms.ExeQuery("seletc * from dbo.watchlist")
count = 0
u = { }
t = []

u_s = {}
total = {}
for wsname,url,id,date in resList:
    if total.has_key(wsname):
        total[wsname]+=1
    else:
        total[wsname] = 1
    str_array = url.split('/')
    url = str_array[-1]
    url = url.replace('%2Fcomments','')
    t1 = transfer_time(date)
    if(u_s.has_key(wsname)):
        if(u_s[wsname].has_key(url)):
            u_s[wsname][url].count += 1
            if t1>u_s[wsname][url].lasttime:
                u_s[wsname][url].lasttime = t1
        else:
            i0 = item()
            i0.count = 1
            i0.lasttime = t1
            u_s[wsname][url] = i0
    else:
        i1 = item()
        i1.count = 1
        i1.lasttime = t1
        i1.total = 1
        u_s[wsname] = {}
        u_s[wsname][url] = i1
output = open('file1.txt','w')
sum = 0
for key,dict in u_s.items():
    for key2,value in dict.items():
        ##print key,key2,value.lasttime,value.count
        res = round(float(value.count)/total[key],3)
        line = key+' '+key2+' '+str(value.lasttime)+' '+str(res)
        output.write(line+'\r\n')
        sum+=1

output.close()
print sum