# !/usr/bin/python
# _*_ coding:utf-8 _*_
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
from get_mashup import MSSSQL

def count_api_mashup():
    ms = MSSSQL(host="172.28.4.193", user="sa", pwd="wy9756784750", db="pweb")
    resList = ms.ExeQuery("select wsname from dbo.ws")
    apiAll = []
    mashupAll = []
    ##output1 = open('api.txt','w')
    ##output2 = open('mashup.txt','w')

    for (wsname,) in resList:
        s = wsname.strip().replace(' ','-')
        apiAll.append(s.upper())
        ##output1.write(s.upper()+'\r\n')

    resList1 = ms.ExeQuery("select wsname from dbo.mp")
    for (wsname,) in resList1:
        s = wsname.strip().replace(' ', '-')
        mashupAll.append(s.upper())
        ##output2.write(s.upper()+'\r\n')

    return apiAll,mashupAll



def test():
    t = write_api_mashupToTxt()
    ms = MSSSQL(host="172.28.4.193", user="sa", pwd="wy9756784750", db="pweb")
    resList = ms.ExeQuery("select * from dbo.watchlist")
    for (wsname,url,id,date) in resList:
        str_array = url.split('/')
        if str_array[-1] in t:
            print wsname,url
##test()
def write_api_mashupToTxt():
    apiAll,mashupAll = count_api_mashup()
    api = set()
    mashup = set()
    Not = set()
    input = open('final_file.txt')
    for line in input:
        s = line.strip('\r\n').split(' ')
        if s[1].upper() in apiAll:
            api.add(s[1])
        elif s[1].upper() in mashupAll:
            mashup.add(s[1])
        else:
            Not.add(s[1])
    output1 = open('u_api.txt','w')
    output2 = open('u_mashup.txt','w')
    output3 = open('u_Not.txt','w')
    for i in api:
        output1.write(i+'\r\n')

    for i1 in mashup:
        output2.write(i1+'\r\n')

    for i2 in Not:
        output3.write(i2+'\r\n')

    ##print len(api)
    ##print len(mashup)
    print 'Not',len(Not)
    input.close()
    output1.close()
    output2.close()
    output3.close()
    return Not


##t = write_api_mashupToTxt()
def xiuzheng():
    output = open('name_replace_1.txt','w')
    t = write_api_mashupToTxt()
    print len(t)
    ms = MSSSQL(host="172.28.4.193", user="sa", pwd="wy9756784750", db="pweb")
    resList = ms.ExeQuery("select wsname,APIhome from dbo.mp")
    for (wsname,APIhome) in resList:
        s = APIhome.strip().split('/')
        if s[-2] in t:
            output.write(wsname.lower().strip().replace(' ','-')+' '+s[-2]+' '+'1'+'\r\n')
            t.remove(s[-2])
    print len(t)
    print t
'''
    input  = open('new_file.txt')
    want_to_delete = []
    for line in input:
        s = line.strip('\r\n').split(' ')
        if s[1] in t:
            want_to_delete.append(line)

    print len(want_to_delete)
    print want_to_delete
'''





xiuzheng()

