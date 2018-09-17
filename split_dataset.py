# !/usr/bin/python
# _*_ coding:utf-8 _*_
import numpy as np
from random import  shuffle
from sklearn.model_selection import train_test_split


##for i in range(len(dataset)):
##    print dataset[i]

def split_train_test(file):
    input = open(file)
    dataset = []
    for line in input:
        dataset.append(line)
    shuffle(dataset)


    train_x, test_x = train_test_split(dataset, test_size=0.1, random_state=0)
    ##print len(train_x),train_x[0]
    ##print len(test_x),test_x[0]
    output1 = open('train.txt','w')
    output2 = open('test.txt','w')
    for i in range(len(train_x)):
        output1.write(train_x[i])

    for i in range(len(test_x)):
        output2.write(test_x[i])



split_train_test('file1.txt')



