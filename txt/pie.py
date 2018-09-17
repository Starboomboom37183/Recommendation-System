 #!/usr/bin/python
#-*-coding:utf-8 -*-
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import matplotlib.pyplot as plt

labels = ['<10', '10-20', '20-30', '30-40', '40-50', '>50']
X = [11945, 1910, 1915, 3006, 749, 510]

fig = plt.figure()
plt.pie(X, labels=labels, autopct='%1.2f%%')
plt.title("")

plt.show()
plt.savefig("PieChart.jpg")