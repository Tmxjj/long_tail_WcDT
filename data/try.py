import numpy as np

# 假设你有 n 个 (125, 2) 的 numpy 数组
n = 3  # 举例，n可以是任意值
arrays = [np.random.rand(125, 2) for _ in range(n)]  # 生成 n 个随机数组

# 使用 numpy.hstack 进行堆叠
result = np.hstack(arrays)

# 或者使用 numpy.concatenate
# result = np.concatenate(arrays, axis=1)

print(result.shape)  # 输出 (125, 2*n)
