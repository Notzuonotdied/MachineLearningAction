"""
scikit-learn 库对knn的支持
数据集是iris虹膜数据集
"""

from sklearn import neighbors
from sklearn.datasets import load_iris

# 查看iris数据集
iris = load_iris()

'''
KNeighborsClassifier(n_neighbors=5, weights='uniform', 
                     algorithm='auto', leaf_size=30, 
                     p=2, metric='minkowski', 
                     metric_params=None, n_jobs=1, **kwargs)
n_neighbors: 默认值为5，表示查询k个最近邻的数目
algorithm:   {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’},指定用于计算最近邻的算法，auto表示试图采用最适合的算法计算最近邻
leaf_size:   传递给‘ball_tree’或‘kd_tree’的叶子大小
metric:      用于树的距离度量。默认'minkowski与P = 2（即欧氏度量）
n_jobs:      并行工作的数量，如果设为-1，则作业的数量被设置为CPU内核的数量
查看官方api：http://scikit-learn.org/dev/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
'''
knn = neighbors.KNeighborsClassifier()
# 训练数据集
knn.fit(iris.data, iris.target)
# 训练准确率
score = knn.score(iris.data, iris.target)

# 预测
predict = knn.predict([[0.1, 0.2, 0.3, 0.4]])
# 预测，返回概率数组
predict2 = knn.predict_proba([[0.1, 0.2, 0.3, 0.4]])

print(predict)
print(iris.target_names[predict])
print(predict2)
