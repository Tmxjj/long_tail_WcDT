import pandas as pd

# 读取CSV文件
df = pd.read_csv('data/long_tail_rttc_with_kqv(1).csv')
# 查看前五行数据
print(df.head())

# 查看数据摘要信息
print(df.info())

# 查看数据的基本统计信息
print(df.describe())
