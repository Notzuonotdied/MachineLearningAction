# coding=utf-8
"""
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharrin
"""
from os import listdir
import operator
from numpy import *


def classify0(in_x, data_set, labels, k):
    """
    分类器
    :param in_x:用于分类的输入向量(列表)
    :param data_set:训练样本集
    :param labels:标签向量
    :param k:距离样本最近的k个邻居
    :return:
    """
    # numpy方法
    # shape表示各个维度大小的元组
    # shape[0]表示在0维度上元组的大小
    data_set_size = data_set.shape[0]
    # numpy方法
    # 在对应维度上重复复制in_x，并去除样本集
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set

    """
    平方之后求和，求和完之后开方
    """
    # 对集合进行平方处理
    sq_diff_mat = diff_mat ** 2
    # sum(axis=1)表示对行的元素进行累加。
    sq_distances = sq_diff_mat.sum(axis=1)
    # 对结果集进行开方
    distances = sq_distances ** 0.5
    # 对结果集升序排序之后返回对数组的索引
    sorted_dist_indicies = distances.argsort()

    """
    统计
    """
    # 新建一个字典，用于保存数据
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_indicies[i]]
        # 如果vote_label不存在，就取默认值0
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1),
                                reverse=True)
    # 返回出现次数最多的分类
    return sorted_class_count[0][0]


def file_matrix(filename):
    """
    使用Python解析文本文件。
    该函数的输入为文件名字符串，输出为训练样本矩阵和类标签向量。
    :param filename:文件名
    :return:训练样本矩阵和类标签向量。
    """
    fr = open(filename)
    # 获取文件的行数
    number_of_lines = len(fr.readlines())
    # 创建返回的Numpy矩阵
    return_mat = zeros((number_of_lines, 3))
    # 准备返回的标签向量
    class_label_vector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()  # 去除多余的回车字符
        list_from_line = line.split('\t')  # 使用\t字符作为分隔符将数据拆分成一个列表
        return_mat[index, :] = list_from_line[0:3]
        # 需要注意的是，我们必须明确地通知解释器，告诉它列表中存储的元素值为整型，
        # 否则Python语言会将这些元素当作字符串处理。
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set):
    """
    对数据进行归一化处理，让数据都在[0,1]范围内，方便进行矩阵运算。
    归一化特征值（将数据处理为0~1范围内）
    :param data_set:数据集合
    :return:
    """
    # 每列中的最小值
    min_val = data_set.min(0)
    # 每列中的最大值
    max_val = data_set.max(0)
    # 取值范围
    ranges = max_val - min_val
    # data中的行的数量
    m = data_set.shape[0]
    # 下面两行的逻辑：newValue = (oldValue-min)/(max-min)
    norm_data_set = data_set - tile(min_val, (m, 1))
    # Numpy库中tile()函数将变量内容复制成输入矩阵同样大小的矩阵
    norm_data_set = norm_data_set / tile(ranges, (m, 1))  # 特征值相除
    return norm_data_set, ranges, min_val


def dating_class_test():
    """
    使用部分数据作为测试样本。
    :return: 无
    """
    # 使用10%的数据进行测试
    ho_ratio = 0.90
    # 从文件中读取数据集合
    dating_data_mat, dating_label = file_matrix('datingTestSet2.txt')
    # 将数据进行归一化处理。
    # 由于数据的大小不统一，所以需要进行归一化处理
    norm_mat, ranges, minVal = auto_norm(dating_data_mat)
    # 获取归一化后一维数组的数量
    m = norm_mat.shape[0]
    # 仅仅拿出m * ho_ratio%的数据进行分类
    num_test_vec = int(m * ho_ratio)
    # 错误的数据的数量
    error_count = 0.0
    for i in range(num_test_vec):
        # 分类
        classifier_result = classify0(norm_mat[i, :],  # 输入向量
                                      norm_mat[num_test_vec:m, :],  # 样本数量
                                      dating_label[num_test_vec:m],  # 标签
                                      3)
        # print("the classifier came back with: %d, the real answer is: %d" %
        #       (classifier_result, dating_label[i]))
        if classifier_result != dating_label[i]:
            error_count += 1.0
    print("the total error rate is: %d%%" % (error_count / float(num_test_vec) * 100))
    print("the sum is %d and the error count is: %d" % (m, error_count))


def classify_person():
    result_list = ['可能性为0', '小几率', '大概率']
    percent_tats = float(input("问题1：玩游戏事件所占的百分比?"))
    fly_miles = float(input("问题2：每年获得的飞行常客里程数?"))
    ice_cream = float(input("问题3：每周消费的冰淇淋公升数?"))
    dating_data_mat, dating_labels = file_matrix('datingTestSet2.txt')
    norm_mat, ranges, min_val = auto_norm(dating_data_mat)
    in_arr = array([fly_miles, percent_tats, ice_cream])
    classifierResult = classify0((in_arr - min_val) / ranges,  # 输入向量
                                 norm_mat,  # 样本数量
                                 dating_labels,  # 标签
                                 3)
    print("You will probably like this person: ", result_list[classifierResult - 1])


def img_to_vector(filename):
    """
    将图片转换为一个向量
    :param filename:图片文件名
    :return: 返回图片向量
    """
    return_vec = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vec[0, 32 * i + j] = int(line_str[j])
    return return_vec


def handwriting_class_test():
    """
    手写数字识别系统的测试代码
    :return:无
    """
    hw_labels = []
    # 将digits/trainingDigits目录下的文件全部存到一个列表中
    training_file_list = listdir('digits/trainingDigits')
    # 获取文件的数量
    m = len(training_file_list)
    # 训练矩阵
    training_mat = zeros((m, 1024))
    # 分类
    # 由于文本中的值已经在0和1之间了，所以我们不需要归一化处理了。
    for i in range(m):
        # 文件名
        file_name = training_file_list[i]
        file = file_name.split('.')[0]  # take off .txt
        # 类别
        class_num = int(file.split('_')[0])
        # 将类别添加到标签列表中
        hw_labels.append(class_num)
        training_mat[i, :] = img_to_vector('digits/trainingDigits/%s' % file_name)
    # 测试样本
    test_file_list = listdir('digits/testDigits')
    # 错误率
    error_count = 0.0
    mTest = len(test_file_list)
    for i in range(mTest):
        file_name = test_file_list[i]
        # 将.txt去掉，仅仅需要的是第一个部分
        file = file_name.split('.')[0]
        # 第一个部分中的_前面的数字是文本内显示的内容
        # 第二个是编号
        class_num = int(file.split('_')[0])
        # 图像文本转向量
        vector_under_test = img_to_vector('digits/testDigits/%s' % file_name)
        # 分类
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        # print("the classifier came back with: %d, the real answer is: %d" %
        #       (classifier_result, class_num))
        if classifier_result != class_num: error_count += 1.0
    print("\n the total number of errors is: %d" % error_count)
    print("\n the total error rate is: %f" % (error_count / float(mTest)))


def create_data_set():
    """
    函数作用：返回一个训练数据，一共四个样本
    :return: 数据，labels
    """
    return array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]]), ['A', 'A', 'B', 'B']
