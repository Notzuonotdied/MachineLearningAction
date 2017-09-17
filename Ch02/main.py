import kNN
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from numpy import *

# 创建自定义的数据
group, labels = kNN.create_data_set()
dating_data_mat, dating_labels = kNN.file_matrix('datingTestSet2.txt')

# 计算错误率
kNN.dating_class_test()

# 喜欢一个人的概率
# kNN.classify_person()

# 手写数字识别系统的测试代码
kNN.handwriting_class_test()


def display_image(is_show):
    font = FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")

    fig = plt.figure(figsize=(10, 6))

    plt.xlabel(r"玩视频游戏所耗时间百分比",
               fontsize=20,
               fontproperties=font)
    plt.ylabel(r"每周消费的冰琪淋公升数",
               fontsize=20,
               fontproperties=font)

    ax = fig.add_subplot(111)
    ax.set_title(r"约会网站配对",
                 fontsize=20,
                 fontproperties=font)
    ax.scatter(dating_data_mat[:, 1],
               dating_data_mat[:, 2],
               15.0 * array(dating_labels),
               15.0 * array(dating_labels))
    if is_show:
        plt.show()
