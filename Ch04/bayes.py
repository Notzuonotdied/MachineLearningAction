"""
Created on Oct 19, 2010

@author: Peter
"""
from numpy import *


def load_data_set():
    """
    创建了一些实验样本
    :return:词条切分侯的文档集合，类别标签的集合
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1表示侮辱性言语, 0表示不是
    return postingList, classVec


def create_vocab_list(data_set):
    """
    创建一个包含在所有文档中出现的不重复词的列表
    :param data_set:文档的新词
    :return:返回新的集合
    """
    # 创建一个空集
    vocabSet = set([])
    for document in data_set:
        # 创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def set_of_words2_vec(vocab_list, input_set):
    """
    检查是否出现某一个单词，若出现就在对应位置置1，否则为0
    如果一个都没有出现的话，就输出这些没有的word
    :param vocab_list:词汇表
    :param input_set:某一个文档
    :return:文档向量，向量的每一个元素是1或者0。
    """
    # 创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocab_list)
    for word in input_set:
        # 创建两个集合的并集
        if word in vocab_list:
            returnVec[vocab_list.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def train_NB_0(train_matrix, train_category):
    """
    朴素贝叶斯分类器训练函数。
    :param train_matrix:文档矩阵
    :param train_category:标签向量
    :return:
    """
    # 传进来的文本个数
    numTrainDocs = len(train_matrix)
    # 计算词个数
    numWords = len(train_matrix[0])
    # 标签中的文档数/文档总数
    # 计算文档中属于侮辱性文档的概率（先验概率）0.5
    pAbusive = sum(train_category) / float(numTrainDocs)
    # --------- 采用拉普拉斯平滑 -----------------
    # 利用贝叶斯分类器对文档进行分类时，要计算多个概率的
    # 乘积以获得文档属于某个类别的概率，即计算
    # p(w0|1)p(w1|1)p(w2|1)。
    # 如果其中一个概率值为0，那么最后的乘积也为0。
    # 为降低这种影响，可以将所有词的出现数初始化为1，并将分母初始化为2。
    # 初始化概率词条的概率为1
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    # 计算每个类别中的文档的数目
    for i in range(numTrainDocs):
        # 对于每一个类别，如果词条出现在文档中
        if train_category[i] == 1:
            # 增加该词条的计数值
            p1Num += train_matrix[i]
            # 增加所有词条的计数值
            p1Denom += sum(train_matrix[i])
        else:
            # 否则将该词条的数目除以总词条的数目得到条件概率
            p0Num += train_matrix[i]
            # 增加所有词条的计数值
            p0Denom += sum(train_matrix[i])
    # 为了防止出现下溢出的问题，对乘积取自然对数，
    # 因为通过求对数可以避免下溢出或者浮点数舍入导致
    # 的错误。同时，采用自然对数进行处理不会有任何损失。
    # 对每个元素做除法
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    # 返回每个类别的条件概率
    return p0Vect, p1Vect, pAbusive


def classify_NB(vec2_classify, p0_vec, p1_vec, p_class1):
    """
    朴素贝叶斯分类函数
    :param vec2_classify:要分类的向量
    :param p0_vec:词条不属于文档的概率
    :param p1_vec:词条属于文档的概率
    :param p_class1:文档属于侮辱类文档的概率
    :return:侮辱类概率大于非侮辱类的时候返回1，反之返回0。
    """
    # 元素相乘
    p1 = sum(vec2_classify * p1_vec) + log(p_class1)
    p0 = sum(vec2_classify * p0_vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def bag_of_words2_vec_MN(vocab_list, input_set):
    """
    朴素贝叶斯词袋模型
    :param vocab_list:词汇表
    :param input_set:某一个文档
    :return:文档向量
    """
    returnVec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            returnVec[vocab_list.index(word)] += 1
    return returnVec


def testing_NB():
    listOPosts, listClasses = load_data_set()
    myVocabList = create_vocab_list(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(set_of_words2_vec(myVocabList, postinDoc))
    p0V, p1V, pAb = train_NB_0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(set_of_words2_vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classify_NB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(set_of_words2_vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classify_NB(thisDoc, p0V, p1V, pAb))


def text_parse(big_string):
    """
    文件解析
    接受一个大字符串并将其解析为字符串列表。
    该函数去掉少于两个字符的字符串，并将所有字符串转换为小写。
    :param big_string: 大的字符串
    :return: 词列表
    """
    import re
    listOfTokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spam_test():
    """
    完整的垃圾邮件测试函数
    对贝叶斯垃圾邮件分类器进行自动化处理。
    导入文件夹spam与ham下的文本文件，并将它们解析为词列表
    :return: 无
    """
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 导入并解析文本文件
        wordList = text_parse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = text_parse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = create_vocab_list(docList)
    trainingSet = range(50)
    testSet = []
    # 留存交叉验证
    for i in range(10):
        # 随机构建训练集
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del list((trainingSet))[randIndex]
    trainMat = []
    trainClasses = []
    # 训练
    for docIndex in trainingSet:
        trainMat.append(bag_of_words2_vec_MN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = train_NB_0(array(trainMat), array(trainClasses))
    errorCount = 0
    # 对测试集分类
    for docIndex in testSet:
        wordVector = bag_of_words2_vec_MN(vocabList, docList[docIndex])
        # 如果邮件分类错误，则错误数加1，最后给出总的错误百分比。
        if classify_NB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("分类错误：", docList[docIndex])
    print('邮件分类的错误率为: ', float(errorCount) / len(testSet))


def calc_most_freq(vocab_list, full_text):
    """
    RSS源分类器及高频词去除函数
    该函数遍历词汇表中的每个词并统计它在文本中出现的次数，
    然后根据出现次数从高到低对词典进行排序，最后返回排序最高的30个单词。
    :param vocab_list:
    :param full_text:
    :return:返回前30个高频单词
    """
    import operator
    freqDict = {}
    for token in vocab_list:
        # 计算每个单词出现的次数
        freqDict[token] = full_text.count(token)
        # 逆序排序
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def local_words(feed1, feed0):
    """
    导入文RSS源，并将它们解析为词列表
    :param feed1:RSS源
    :param feed0:RSS源
    :return:
    """
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        # 每次访问一条RSS源
        wordList = text_parse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = text_parse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 去掉出现次数最高的那些词
    vocabList = create_vocab_list(docList)
    # 这里采用了停用词，可以查看地址：http://www.ranks.nl/stopwords
    stopWordList = stop_words()
    for stopWord in stopWordList:
        if stopWord in vocabList:
            vocabList.remove(stopWord)
    # top30Words = calc_most_freq(vocabList, fullText)
    # for pairW in top30Words:
    #     if pairW[0] in vocabList:
    #         vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(5):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    # 训练分类器
    for docIndex in trainingSet:
        trainMat.append(bag_of_words2_vec_MN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = train_NB_0(array(trainMat), array(trainClasses))
    errorCount = 0
    # 分类剩下的东西
    for docIndex in testSet:
        wordVector = bag_of_words2_vec_MN(vocabList, docList[docIndex])
        if classify_NB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def get_top_words(ny, sf):
    """
    最具表征性的词汇显示函数
    :param ny:正面
    :param sf:反面
    :return:
    """
    vocabList, p0V, p1V = local_words(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF==============>%f" % len(sortedSF))
    # for item in sortedSF:
    #     print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY==============>%f" % len(sortedNY))
    # for item in sortedNY:
    #     print(item[0])


def stop_words():
    import re
    # 详情请见：http://www.ranks.nl/stopwords
    wordList = open('stoptxt.txt').read()
    listOfTokens = re.split(r'\W*', wordList)
    return [tok.lower() for tok in listOfTokens]
