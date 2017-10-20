"""
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
"""
from numpy import *


def loadDataSet():
    """
    加载数据
    :return:返回数据集合和数据标签集合
    """
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 回归系数、X1、X2
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    """
    Sigmoid函数
    :param inX: X
    :return: 返回Sigmoid的函数值
    """
    return longfloat(1.0 / (1 + exp(-inX)))


def gradAscent(dataMatIn, classLabels):
    """
    梯度上升算法
    :param dataMatIn:每列分别代表每个不同的特征，每行则代表每个训练样本
    :param classLabels:分类标签
    :return:
    """
    # 转换为NumPy矩阵数据类型
    dataMatrix = mat(dataMatIn)  # 转换成NumPy矩阵
    # 转换成NumPy矩阵，将行变成列
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    # 向目标移动的步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    weights = ones((n, 1))
    # for循环结束后将返回训练好的回归系数
    for k in range(maxCycles):
        # 矩阵相乘
        # 计算整个数据集的梯度
        h = sigmoid(dataMatrix * weights)
        # 以下两行计算真实类别与预测类别的差值
        error = (labelMat - h)  # 向量相减
        weights = weights + alpha * dataMatrix.transpose() * error  # 向量相乘

    return weights


def plotBestFit(weights):
    """
    画出数据集和logistic回归最佳拟合直线的函数
    :param weights:
    :return:
    """
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = list(shape(dataArr))[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # 最佳拟合直线
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y.transpose())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    """
    随机梯度上升算法
    :param dataMatrix:数据集合
    :param classLabels:分类
    :return:返回系数
    """
    m, n = shape(dataMatrix)
    alpha = 0.01
    # 初始化权重为1
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    改进的随机梯度上升算法
    :param dataMatrix:数据集
    :param classLabels:数据标签
    :param numIter:迭代次数
    :return:返回系数
    """
    m, n = shape(dataMatrix)
    weights = ones(n)  # initialize to all ones
    for j in list(range(numIter)):
        dataIndex = range(m)
        for i in range(m):
            # alpha每次迭代时需要调整。
            ######################################
            # 在降低alpha的函数中，alpha每次减少1/(j+i) ，
            # 其中j是迭代次数，i是样本点的下标1 。
            # 这样当j<<max(i)时，alpha就不是严格下降的。
            # 避免参数的严格下降也常见于模拟退火算法等其他优化算法中。
            ######################################
            alpha = 4 / (1.0 + j + i) + 0.0001
            # 随机选取更新
            # go to 0 because of the constant
            ######################################
            # 这里通过随机选取样本来更新回归系数。这种方法将减少周期性的波动
            ######################################
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del list(dataIndex)[randIndex]
    return weights


def classifyVector(inX, weights):
    """
    logistic回归分类函数
    :param inX:特征向量
    :param weights:回归系数
    :return:返回分类标签
    """
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    """
    打开测试集和训练集，并对数据进行格式化处理的函数
    :return:错误率
    """
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    """
    调用函数colicTest()10次并求结果的平均值。
    :return: 错误率
    """
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (
        numTests, errorSum / float(numTests)
    ))
