# !/usr/bin/python
# _*_ coding:utf-8 _*_
import pymssql
import xlrd
from get_mashup import MSSSQL
ms = MSSSQL(host="172.27.134.219", user="sa", pwd="wy9756784750", db="pweb")
resList = ms.ExeQuery("select * from dbo.watchlist")
count = 0
u = { }
for wsname,url,id,date in resList:
    if u.has_key(wsname):
        u[wsname]+=1
    else:
        u[wsname] = 1
sum = [0,0,0,0,0,0]
for name,count in u.items():
    if count>50:
        ##print name,count
        sum[5]+=1
    elif count>40 and count<=50:
        sum[4]+=1
    elif count>30 and count<=40:
        sum[3]+=1
    elif count>20 and count<=30:
        sum[2]+=1
    elif count>10 and count<=20:
        sum[1]+=1
    else:
        sum[0]+=1
print sum
