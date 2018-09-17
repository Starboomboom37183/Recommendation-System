# !/usr/bin/python
# _*_ coding:utf-8 _*_
from get_mashup import MSSSQL
import numpy as np
import datetime
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
    d1= datetime.datetime.strptime(str, '%Y-%m-%d %H:%M:%S')
    return d1

input1 = open('User.txt')
t_user = {}
for line in input1:
    s = line.strip('\r\n').split(' ')
    t_user[s[0]] = datetime.datetime.strptime('2017-09-03 17:00:02','%Y-%m-%d %H:%M:%S')
input2 = open('webservice.txt')
t_s = {}
for line in input2:
    s = line.strip('\r\n').split(' ')
    t_s[s[0]] = datetime.datetime.strptime('2017-09-03 17:00:02','%Y-%m-%d %H:%M:%S')
input1.close()
input2.close()
'''
ms = MSSSQL(host="172.28.37.29", user="sa", pwd="wy9756784750", db="pweb")
resList = ms.ExeQuery("select title,updated from dbo.userinfo")
c = 0

for title,updated in resList:
    if t_user.has_key(title):
        updated = updated.replace('T',' ')
        updated = updated.replace('Z', '')
        d2 = datetime.datetime.strptime(updated, '%Y-%m-%d %H:%M:%S')
        t_user[title] = d2

input3 = open('final_file.txt')
d = datetime.datetime.strptime('2005-09-03 17:00:02','%Y-%m-%d %H:%M:%S')
print d
set_time = {}
c = 0
for line in input3:
    s = line.strip('\r\n').split(' ')
    t = d + datetime.timedelta(seconds=int(s[2]))
    ut = (t-t_user[s[0]]).days
    if ut<0:
        ut = 0
'''
input5 = open('name_replace.txt')
rep = {}
for line in input5:
    s = line.strip('\r\n').split(' ')
    rep[s[1]] = s[0]

ms = MSSSQL(host="172.28.37.29", user="sa", pwd="wy9756784750", db="pweb")
resList = ms.ExeQuery("select * from dbo.watchlist")
c = 0
for wsname,url,id,date in resList:
    str_array = url.split('/')
    url = str_array[-1]
    url = url.replace('%2Fcomments', '')
    if rep.has_key(url):
        url = rep[url]
    t = transfer_time(date)
    if t_s.has_key(url):
        if t<t_s[url]:
            t_s[url] = t
    if t_user.has_key(wsname):
        if t<t_user[wsname]:
            t_user[wsname] = t


##print t_user
##print t_s
input3 = open('final_file.txt')
d = datetime.datetime.strptime('2005-09-03 17:00:02','%Y-%m-%d %H:%M:%S')
print d
set_time = {}
c = 0
output = open('final_file_1.txt','w')
for line in input3:
    s = line.strip('\r\n').split(' ')
    t = d + datetime.timedelta(seconds=int(s[2]))
    ut = (t-t_user[s[0]]).days
    st  = (t-t_s[s[1]]).days
    ##print ut,st
    '''
    if ut == 0:
        print s[0],s[1],t_user[s[0]],t
    '''
    line = line.strip('\r\n')+' '+str(ut)+' '+ str(st)+'\r\n'
    output.write(line)



