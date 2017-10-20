from numpy import array

from Ch05 import logRegres

dataArr, labelMat = logRegres.loadDataSet()
weight = logRegres.gradAscent(dataArr, labelMat)
print(weight)

# 画出数据集和logistic回归最佳拟合直线的函数
# logRegres.plotBestFit(weight)

# 使用随机梯度上升算法
# weights=logRegres.stocGradAscent0(array(dataArr),labelMat)
# logRegres.plotBestFit(weights)

# 使用改进的随机梯度上升算法
# weights = logRegres.stocGradAscent1(array(dataArr), labelMat, 500)
# logRegres.plotBestFit(weights)

# 从疝气病症预测病马的死亡率
logRegres.multiTest()