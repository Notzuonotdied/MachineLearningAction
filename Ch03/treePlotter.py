"""
Created on Oct 14, 2010

@author: Peter Harrington
"""
import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def get_num_leafs(my_tree):
    """
    获取叶节点的数目和树的层数
    :param my_tree: 树
    :return:返回叶节点的数目与和树的层数
    """
    numLeafs = 0
    firstSides = list(my_tree.keys())
    firstStr = firstSides[0]
    secondDict = my_tree[firstStr]
    for key in secondDict.keys():
        # 测试节点的数据类型是否为字典
        if type(secondDict[
                    key]).__name__ == 'dict':
            numLeafs += get_num_leafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def get_tree_depth(my_tree):
    """
    计算树的深度
    :param my_tree:树
    :return:返回深度
    """
    maxDepth = 0
    firstSides = list(my_tree.keys())
    firstStr = firstSides[0]
    secondDict = my_tree[firstStr]
    for key in secondDict.keys():
        # test to see if the nodes are dictonaires, if not they are leaf nodes
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + get_tree_depth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plot_node(node_txt, center_pt, parent_pt, node_type):
    """
    绘制带箭头的注解
    :param node_txt:
    :param center_pt:
    :param parent_pt:
    :param node_type:
    :return:
    """
    create_plot.ax1.annotate(node_txt,
                             xy=parent_pt,
                             xycoords='axes fraction',
                             xytext=center_pt,
                             textcoords='axes fraction',
                             va="center",
                             ha="center",
                             bbox=node_type,
                             arrowprops=arrow_args)


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    """
    通过计算父节点和子节点的中间位置，并在此处添加简单的文本信息
    :param cntr_pt:子节点
    :param parent_pt:父节点
    :param txt_string:填充的文本内容
    :return:无
    """
    xMid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    yMid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(xMid,
                         yMid,
                         txt_string,
                         va="center",
                         ha="center",
                         rotation=30)


def plot_tree(my_tree, parent_pt, node_txt):  # if the first key tells you what feat was split on
    """
    绘制树
    :param my_tree:树
    :param parent_pt:父节点
    :param node_txt:节点信息
    :return:无
    """
    # 计算叶子的数量
    numLeafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    firstSides = list(my_tree.keys())
    firstStr = firstSides[0]
    cntrPt = (plot_tree.xOff + (1.0 + float(numLeafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    # 标记子节点属性值
    plot_mid_text(cntrPt, parent_pt, node_txt)
    plot_node(firstStr, cntrPt, parent_pt, decisionNode)
    secondDict = my_tree[firstStr]
    # 减少y偏移
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in secondDict.keys():
        # test to see if the nodes are dictonaires, if not they are leaf nodes
        if type(secondDict[key]).__name__ == 'dict':
            plot_tree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(secondDict[key], (plot_tree.xOff, plot_tree.yOff), cntrPt, leafNode)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntrPt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


def create_plot(in_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    # 树的宽度
    plot_tree.totalW = float(get_num_leafs(in_tree))
    # 树的深度
    plot_tree.totalD = float(get_tree_depth(in_tree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()


# def createPlot():
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
#    plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
#    plotNode('叶子节点', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()

def retrieve_tree(i):
    """
    为了节省大家的时间，函数retrieveTree输出预先存储的树信息，避免了每次测试代码时都要从数据中创建树的麻烦。
    :param i:树的下标
    :return:树
    """
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]
