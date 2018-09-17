# !/usr/bin/python
# _*_ coding:utf-8 _*_
import numpy as np
from get_mashup import MSSSQL

def get_user():
    ms = MSSSQL(host="172.28.4.193", user="sa", pwd="wy9756784750", db="pweb")
    resList = ms.ExeQuery("select * from dbo.watchlist")
    user_name = {}
    for wsname,url,id,date in resList:
        if user_name.has_key(wsname):
            user_name[wsname]+=1
        else:
            user_name[wsname] = 1
    s_name = []
    count = 0
    for key,value in user_name.items():
        if value>50:
            ##print key,value
            s_name.append(key)
            count+=1

    ##print count
    ##print len(s_name)
    return s_name




def make_dataset(fileName):
    u_name = {}
    input = open(fileName)
    for line in input:
        s = line.strip('\r\n').split(' ')
        ##print float(s[-1])*1000
        if(u_name.has_key(s[0])):
            u_name[s[0]]+=1
        else:
            u_name[s[0]] = 1

    name_set = []
    for key,value in u_name.items():
        if value>=50:
            name_set.append(key)


    ##print len(name_set)


##make_dataset('file1.txt')
name = []
name = get_user()
input = open('file1.txt')
input2 = open('name_replace.txt')
rep = {}
for line in input2:
    s = line.strip('\r\n').split(' ')
    rep[s[1]] = s[0]

##print rep
output = open('new_file.txt','w')
for line in input:
    line = line.replace('%2Fcomments','')
    s = line.strip('\r\n').split(' ')

        ##print s[1]
    if s[0] in name:
        line1 = ''
        if rep.has_key(s[1]):
            line1 = s[0]+' '+rep[s[1]]+' '+s[2]+' '+s[3]+'\r\n'
        else:
            line1 = line
        output.write(line1)

output.close()

