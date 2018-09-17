# !/usr/bin/python
# _*_ coding:utf-8 _*_
import numpy as np

input = open('u_Not.txt')
Not = []
for line in input:
    s = line.strip('\r\n')
    print s
    Not.append(s)

print len(Not)

input2 = open('new_file.txt')
output = open('final_file.txt','w')
name = set()
user = set()
count = 0
count1 = 0
for line in input2:
    count1+=1
    s = line.strip('\r\n').split(' ')
    if s[1] in Not:
        print line
        count+=1
        name.add(s[1])
    else:
        output.write(line)
        user.add(s[0])
input2.close()
output.close()

print len(name)
print count
print (count1-count)
print len(user)