# !/usr/bin/python
# _*_ coding:utf-8 _*_
import time
import random
import string

class Student(object):
	def __init__(self, name, gender, age):
		self.__name = name
		self.__gender = gender
		self.__age = age

	# 取得age属性
	def getAge(self):
		return self.__age

	# 打印
	def printStudent(self):
		return self.__name, self.__gender, self.__age

# 生成包含随机学生对象的list
def generateStudent(num):
	# num为需要生成的测试对象数
	list = []
	for i in range(num):
		randName = ''.join(random.sample(string.ascii_letters, 4))
		randGender = random.choice(['Male', 'FeMale'])
		randAge = random.randint(10,30)
		s = Student(randName, randGender, randAge)
		list.append(s)
	return list

# 冒泡排序
def sortStudent(list):
	for i in range(len(list)):
		for j in range(1, len(list)-i):
			if list[j-1].getAge() > list[j].getAge():
				list[j-1], list[j] = list[j], list[j-1]
	return list

# def ageReturn(list):
# 	return list.age

if __name__ == '__main__':

    list = generateStudent(5)
    for j in range(len(list)):
        print(list[j].printStudent())

	sorted(list,key=lambda student: student.getAge()) # 将对象的属性作为排序的Key
    pairs = [('one', 1), ('two', 2), ('three', 3), ('five', 5), ('zero', 0), ('four', 4)]
    pairs = sorted(pairs, key=lambda pair: pair[1])
    for j in range(len(pairs)):
        print pairs[j]
	# 方法2中，使用使用1000000个测试数据的排序时间是0.575秒。虽然不是很精确，但差别显而易见了。