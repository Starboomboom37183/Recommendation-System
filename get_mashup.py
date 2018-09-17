# !/usr/bin/python
# _*_ coding:utf-8 _*_
import pymssql
import xlrd


class MSSSQL:
    def __init__(self, host, user, pwd, db):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.db = db

    def __GetConnect(self):
        if not self.db:
            raise (NameError, "没有数据库")
        self.conn = pymssql.connect(host=self.host, user=self.user, password=self.pwd, database=self.db)
        cur = self.conn.cursor()

        if not cur:
            raise (NameError, "连接数据库失败")
        else:
            return cur

    def ExeQuery(self, sql):
        cur = self.__GetConnect()
        cur.execute(sql)
        resList = cur.fetchall()

        self.conn.close()
        return resList

    def ExeNonQuery(self, sql):

        cur = self.__GetConnect()
        cur.execute()
        self.conn.commit()
        self.conn.close()


def main():
    data = xlrd.open_workbook('Composition_mashup.xlsx')

    table = data.sheets()[0]

    nrows = table.nrows

    dict = {}
    for i in range(1, nrows):
        str = table.row_values(i)[0][8:]
        list1 = table.row_values(i)[1].split(',')

        dict[str] = list1

    print "你好，开始"
    ms = MSSSQL(host="172.28.4.193", user="sa", pwd="wy9756784750", db="pweb")
    resList = ms.ExeQuery("SELECT wsname,rating FROM dbo.mp")
    count = 0
    for (wsname, rating) in resList:
        if dict.has_key(wsname):
            print wsname, dict[wsname]
            count += 1


    print count
    print "你好，结束"


if __name__ == '__main__':
    main()
