import numpy as np

N = 6
n = 2

# 随机生成矩阵
A11 = np.random.rand(N, N)
A22 = np.random.rand(N, N)
A33 = np.random.rand(n, n)
A13 = np.random.rand(N, n)
A23 = np.random.rand(N, n)
A31 = np.random.rand(n, N)
A32 = np.random.rand(n, N)

# 随机生成已知解向量
u1 = np.random.rand(N)
u2 = np.random.rand(N)
u3 = np.random.rand(n)

# 计算右侧向量
f1 = A11 @ u1 + A13 @ u3
f2 = A22 @ u2 + A23 @ u3
f3 = A31 @ u1 + A32 @ u2 + A33 @ u3

with open("schur_input_large.txt", "w") as f:
    f.write(f"{N} {n}\n")
    for matrix in [A11, A22, A33, A13, A23, A31, A32]:
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")
    for vector in [f1, f2, f3]:
        f.write(" ".join(map(str, vector)) + "\n")

with open("schur_solution.txt", "w") as f:
    f.write(f"{u1}\n")
    f.write(f"{u2}\n")
    f.write(f"{u3}\n")

