'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
import operator
from math import log


def create_data_set():
    """
    创建数据集合
    :return: 返回创建好的数据集合
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


def calc_shannon_ent(data_set):
    """
    计算给定数据集合的香农熵
    :param data_set:数据集
    :return:返回香农熵
    """
    numEntries = len(data_set)
    labelCounts = {}
    # 为所有可能分类创建字典
    for featVec in data_set:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 初始化香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        # 以2为底数求对数
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def split_data_set(data_set, axis, value):
    """
    划分数据集合
    :param data_set:待划分的数据集合
    :param axis:按某个轴划分，划分数据集合的特征
    :param value:需要返回特征的值
    :return:
    """
    retDataSet = []
    for featVec in data_set:
        if featVec[axis] == value:
            # 抽取数据
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def choose_best_feature_to_split(data_set):
    """
    选择最好的数据集合划分方式
    :param data_set: 待划分的数据集合
    :return: 返回最好的用于划分数据集合的特征
    """
    # 数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签。
    numFeatures = len(data_set[0]) - 1
    baseEntropy = calc_shannon_ent(data_set)
    bestInfoGain = 0.0
    bestFeature = -1
    # 遍历所有的属性
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        featList = [example[i] for example in data_set]
        uniqueVals = set(featList)
        # 初始化香农熵
        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        # 信息增益是熵的减少或者是数据无序度的减少
        for value in uniqueVals:
            subDataSet = split_data_set(data_set, i, value)
            prob = len(subDataSet) / float(len(data_set))
            newEntropy += prob * calc_shannon_ent(subDataSet)

        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            # 计算最好的信息熵
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回最好的划分数据集合的特征


def majority_cnt(class_list):
    """
    如果数据集已经处理了所有属性，但是类标签依然不是唯一的，
    此时我们需要决定如何定义该叶子节点，在这种情况下，我们
    通常会采用多数表决的方法决定该叶子节点的分类。
    :param class_list:分类列表
    :return:返回可能性最大的一个类别
    """
    classCount = {}
    for vote in class_list:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]


def create_tree(data_set, labels):
    """
    创建树的函数代码
    :param data_set:数据集合
    :param labels:标签列表，标签列表包含了数据集中所有特征的标签
    :return:返回树
    """
    classList = [example[-1] for example in data_set]
    # 类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的
    if len(data_set[0]) == 1:
        return majority_cnt(classList)
    bestFeat = choose_best_feature_to_split(data_set)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    # 得到包含的所属属性值
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in data_set]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # copy all of labels, so trees don't mess up existing labels
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = create_tree(
            split_data_set(data_set, bestFeat, value),
            subLabels)
    # 字典变量myTree存储了树的所有信息
    return myTree


def classify(input_tree, feat_labels, test_vec):
    """
    使用决策树的分类函数
    :param input_tree: 输入树
    :param feat_labels:特征标签
    :param test_vec:测试向量集合
    :return:分类后的标签
    """
    firstSides = list(input_tree.keys())
    firstStr = firstSides[0]  # 找到输入的第一个元素
    secondDict = input_tree[firstStr]
    # 将标签字符串转换为索引
    featIndex = feat_labels.index(firstStr)
    key = test_vec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        # 递归遍历整棵树，比较test_vec变量中的值与树节点中的值，
        # 如果到达叶子节点，则返回当前节点的分类标签
        classLabel = classify(valueOfFeat, feat_labels, test_vec)
    else:
        classLabel = valueOfFeat
    return classLabel


def store_tree(input_tree, filename):
    """
    使用决策树存储
    :param input_tree:要存储的树
    :param filename:存储的文件的名称
    :return:无
    """
    # 序列化对象存储
    import pickle
    fw = open(filename, 'w')
    pickle.dump(input_tree, fw)
    fw.close()


def grab_tree(filename):
    """
    使用pickle获取存储的树
    :param filename:
    :return:
    """
    import pickle
    fr = open(filename)
    return pickle.load(fr)
