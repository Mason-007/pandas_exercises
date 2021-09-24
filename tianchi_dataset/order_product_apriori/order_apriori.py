import pandas as pd
from efficient_apriori import apriori

# 加载数据
# df_product = pd.read_csv('./product.csv')
# df_date = pd.read_csv('./date.csv')
df_order = pd.read_csv('./order.csv' , encoding='gbk')
# df_customer = pd.read_csv('./customer.csv')

# print(df_order.info())
# 创建一个一维数组, index为客户id, key为产品名称
order_series = df_order.set_index('客户ID')['产品名称']
order_series.sort_index(inplace=True)
# print(order_series)

# 对数据集进行格式转换
trans = []
temp_index = 0
for i , v in order_series.items():
    # print(f'{i}: {v}')
    if i != temp_index:
        temp_set = set()
        temp_index = i
        temp_set.add(v)
        trans.append(temp_set)
    else: 
        temp_set.add(v)

# (A=>B) support = P(A,B) confidence = P(B|A) lift = c / P(B)
freq_set , rule = apriori(trans, min_support=0.04, min_confidence=0.3)
print(f"频繁项集:\n{freq_set}\n\n")
print(f"关联规则:\n{rule}\n\n")
        



