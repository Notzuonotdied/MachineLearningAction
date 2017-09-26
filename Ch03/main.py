from Ch03 import treePlotter
from Ch03 import trees

# 创建数据集合
data, label = trees.create_data_set()
print("数据集合：", data)
print("数据标签：", label)

# 计算香农熵
print("香农熵：", trees.calc_shannon_ent(data))

# 香农熵越高，则混合的数据也越多
data[0][-1] = '香农熵'
print("修改后的数据：", data)
print("香农熵：", trees.calc_shannon_ent(data))

# 数据划分测试
print("划分1：", trees.split_data_set(data, 0, 1))
print("划分2：", trees.split_data_set(data, 0, 0))

# 选择最好的数据集合划分方式
print("第", trees.choose_best_feature_to_split(data),
      "个特征是最好的用于划分数据集合的特征。")

# 创建决策树
myTree = trees.create_tree(data, label)
print(myTree)
# 显示决策树
# treePlotter.create_plot(myTree)
print("树的深度：", treePlotter.get_tree_depth(myTree))
print("树的叶子节点数：", treePlotter.get_num_leafs(myTree))

_, labels = trees.create_data_set()
myTree = treePlotter.retrieve_tree(0)
print(trees.classify(myTree, labels, [1, 0]),
      trees.classify(myTree, labels, [1, 1]))

# 使用决策树预测隐形眼睛类型
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.create_tree(lenses, lensesLabels)
treePlotter.create_plot(lensesTree)
