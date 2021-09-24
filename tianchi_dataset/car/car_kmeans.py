from operator import concat
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans



data = pd.read_csv('./car_price.csv')
# 展示数据
# print(data.head())


car_info = data[['car_ID' , 'CarName']]
# 保存车名和id 
# print(car_info.head())

# 对结果无影响，删除车名和id
data = data.drop(['car_ID','CarName'] , axis=1)

# print(data.info())
# 将数值型和字符串分开
# 将数值型特征按比例缩放的一定范围内

data_str = data.select_dtypes(include=object)
data_num = data.select_dtypes(exclude=object)

# 标准化标签
data_str = data_str.apply(LabelEncoder().fit_transform)
# 合并标准化的数据
train_x = pd.concat((data_str , data_num) , axis=1)
# 归一化到[0,1]
train_x = preprocessing.MinMaxScaler().fit_transform(train_x)
# print(train_x)

# 寻找k值,用手肘法计算不同k取值的误差平方和(sse)
sse = []
n = 50
for k in range(1,n):
    kmeans = KMeans(n_clusters=k) #n_clusters聚类数
    kmeans.fit(train_x) #喂数拟合
    sse.append(kmeans.inertia_) #计算簇内sse
    
x = range(1,n)
plt.xlabel('k')
plt.ylabel('sse')
plt.plot(x,sse,'o-') #todo
# plt.show() 

# 使用手肘法确定k值后，开始聚类分析
kmeans = KMeans(n_clusters = 8)
kmeans.fit(train_x)
labels = kmeans.predict(train_x) # 按照已存在的质心对所以数据进行聚类
# print(labels)

# 将聚类结果和之前删去的车名表合并
result = pd.concat((car_info , pd.DataFrame(labels)) , axis = 1)
result.rename({0:u'Clusters'} , axis = 1 , inplace=True)
result.to_csv('./cluster_result.csv' , index = False)
# print(result)

# 选出名字为'volkswagen'或者‘vm'的行
vm = result[result.CarName.str.contains('volkswagen|vm')]
# 查看它们属于哪些组
list_vm = vm.Clusters.drop_duplicates().tolist()
# print(list_vm)

for i in list_vm:
    vm_name = vm[vm['Clusters'] == i]['CarName'].drop_duplicates().tolist()
    competitors = result[result['Clusters'] == i]['CarName'].drop_duplicates() #每个分组的车型
    vm_competitors = competitors[~competitors.str.contains('vm|volkswagen')].tolist() #每组竞争车型
    print(f"“大众汽车”{vm_name}所在分组{i}的竞争车型有:\n{vm_competitors}\n\n\n")
    








