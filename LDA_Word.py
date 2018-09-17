# coding=utf-8
'''
英文的词干化和去停用词
@author: yongwang260
'''
import nltk
import string
import re
import os
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


class EnPreprocess:

    def __init__(self):
        print'English token and stopwords remove...'


    def FileRead(self, filePath):  # 读取内容
        f = open(filePath)
        raw = f.read()
        return raw
    def SenToken(self,raw):#分割成句子
        sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_tokenizer.tokenize(raw)
        return  sents

    def CleanLines(self, line):#去除标点和数字
        delEStr = string.punctuation + string.digits  # ASCII 标点符号，数字
        space = ''
        for i in range(len(delEStr)):
            space+=' '
        identify = string.maketrans(delEStr, space)
        cleanLine = line.translate(identify)  # 去掉ASCII 标点符号
        return cleanLine

    def WordTokener(self, sent):  # 将单句字符串分割成词
        result = ''
        wordsInStr = nltk.word_tokenize(sent)
        return wordsInStr

    def load_stop(self,file):
        input = open(file)
        stop = []
        for line in input:
             s = line.strip()
             stop.append(s)
        return stop


    def CleanWords(self,wordsInStr):#去掉标点符号，长度小于3的词以及non-alpha词，小写化
        cleanWords=[]
        stop = self.load_stop('LDA_stop.txt')
        for words in wordsInStr:
            cleanWords+= [[w.lower() for w in words if w.lower() not in stop and 3<=len(w)]]
        return cleanWords

    def StemWords(self, cleanWordsList):
        stemWords = []
        for words in cleanWordsList:
            stemWords += [[PorterStemmer().stem(w) for w in words]]

        return stemWords

    def WordsToStr(self, stemWords):
        strLine = []
        for words in stemWords:
            strLine += [w for w in words]
        return strLine
    def WriteResult(self,result,resultPath):
        f=open(resultPath,"w")
        for line in result:
            f.write(str(line))
        f.close()





def Word_process():
    enPre = EnPreprocess()
    input = open('LDA_train.txt')
    ##print stopwords.words('english')
    output = open('LDA_result.txt','w')
    for line in input:
        ##print line
        sents = enPre.SenToken(line.strip('\r\n'))
        ##print sents
        cleanLines = [enPre.CleanLines(line) for line in sents]
        ##print cleanLines
        words = [enPre.WordTokener(cl) for cl in cleanLines]
        ##print words
        cleanWords = enPre.CleanWords(words)
        ##print cleanWords
        stemWords = enPre.StemWords(cleanWords)
        ##print stemWords
        strLine = enPre.WordsToStr(stemWords)
        ##print strLine
        ##print str(strLine)
        ##Lines.append(strLine)
        s = ''
        for word in strLine:
            ##print word
            s+=str(word)+' '
        s.strip()
        output.write(s+'\r\n')

    ##print Lines

    output.close()
    print 'jieshu'


Word_process()