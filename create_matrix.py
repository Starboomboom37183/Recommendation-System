# !/usr/bin/python
# _*_ coding:utf-8 _*_
import numpy as np
'''
input1 = open('u_api.txt')
input2 = open('u_mashup.txt')
output = open('webservice.txt','w')
count = 0
for line in input1:
    s = line.strip('\r\n')
    s = s+' '+str(count)
    output.write(s+'\r\n')
    count+=1
##0-2712为api
print count
for line in input2:
    s = line.strip('\r\n')
    s = s + ' ' + str(count)
    output.write(s + '\r\n')
    count+=1

print count

input1.close()
input2.close()
output.close()
'''
input3 = open('final_file.txt')
output2 = open('User.txt','w')

user = {}
user_find = {}
count = 0
for line in input3:
    s = line.strip('\r\n').split(' ')
    if user.has_key(s[0]):
        continue
    else:
        user[s[0]] = count
        user_find[count] = s[0]
        output2.write(s[0]+' '+str(count)+'\r\n')
        count+=1

print user
print count
input3.close()
ws = {}
ws_find = {}
input1 = open('webservice.txt')
for line in input1:
    s = line.strip('\r\n').split(' ')
    ##print s[0],s[1]
    ws[s[0]] = int(s[1])
    ws_find[int(s[1])] = s[0]

matrix = np.zeros((510,4813))
##print matrix.dtype
##print matrix
input3 = open('final_file.txt')
c = 0
for line in input3:
   s = line.strip('\r\n').split(' ')
   i = user[s[0]]
   j = ws[s[1]]
   if(matrix[i][j]!=0):
       c+=1
       matrix[i][j]+=float(s[3])
       print s[0],s[1]
   ##print s[3]
   else:
       matrix[i][j] = float(s[3])
##print c
##print np.sum(matrix!=0)+np.sum(matrix==0)
##print (1-float(np.sum(matrix!=0))/2454630)

'''
out = open('matrix.txt','w')
row,col =  matrix.shape
for i in range(row):
    line = ''
    for j in range(col):
        line = line+str(matrix[i][j])+' '
    line +=''
    out.write(line)
out.close()
'''








