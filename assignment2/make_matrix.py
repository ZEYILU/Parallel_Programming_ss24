import numpy as np

# 设置随机种子以保证重复性
np.random.seed(0)

# 生成100x150的矩阵
A = np.random.rand(200, 150) * 10  # 范围从0到10

# 保存到文件
np.savetxt('matrixA.txt', A, fmt='%.2f', delimiter=',')

# 设置随机种子以保证重复性
np.random.seed(1)

# 生成150x100的矩阵
B = np.random.rand(150, 300) * 10  # 范围从0到10

# 保存到文件
np.savetxt('matrixB.txt', B, fmt='%.2f', delimiter=',')
