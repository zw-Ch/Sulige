# -*- codeing = utf-8 -*-
# @Time : 2022/9/19 11:56
# @Author : 罗佩瑶
# @File : 138-845A.py
# @Software : PyCharm


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
import seaborn as sns # 可视化库
from scipy import stats, integrate
import pandas as pd
from scipy.stats import *
#实例化font_manager
songti = font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
from matplotlib.pyplot import plot,savefig



###1.读取数据文件1
object1= open('D:\\mlp-ledo-new data\\138\\138-845A\\138-845A_hldt.dat','r')
data1 = []
for line in object1:
    s = line.strip().split('\t')#strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）
    data1.append(s)
columns1 = data1[4:5][0]#属性名
data1=data1[6:]
#print(data1)


###2.读取数据文件2
object2 = open('D:\\mlp-ledo-new data\\138\\138-845A\\138-845A_sdt-ls.dat','r')
data2 = []
for line in object2:
    s = line.strip().split('\t')
    data2.append(s)
columns2 = data2[4:5][0]
data2 = data2[6:]


###3.写成dataframe格式
df_data1 = pd.DataFrame(data1,columns=columns1)
df_data2 = pd.DataFrame(data2,columns=columns2)

df_data1.drop(len(df_data1)-1,inplace=True)#本data1最后两行有空值
#drop函数默认删除行，列需要加axis = 1
#inplace=True，那么原数组直接就被替换
df_data2.drop(len(df_data2)-1,inplace=True)
print(df_data1)
print(df_data2)

###4.获取密度数据和波速数据
list1 = []
list2 = []
for i in df_data1['RHOB']:
    list1.append(float(i))
for j in df_data2['VP1']:
    list2.append(float(j))
print(list1)
print(list2)


###5.获取深度数据
depth1 = []
depth2 = []
for i in df_data1['DEPTH_WMSF']:
    depth1.append(float(i))
for j in df_data2['DEPTH_WMSF']:
    depth2.append(float(j))
print(depth1)
print(depth2)


'''
print(len(list1))#3987
print(len(list2))#3972
print(depth1)
print(depth2)
print(len(depth1))#2643
print(len(depth2))#3373
'''


###6.数据按深度对齐
# 将list1与list2数据开头按深度对齐 + 结尾深度对齐
num1 = depth1.index(90.0684)
print(num1) #0
num2 = depth2.index(90.0684)
print(num2) #148

num_ending1 = depth1.index(272.3388)
num_ending2 = depth2.index(272.3388)
print(num_ending1) #3823
print(num_ending2) #3971

if (num_ending1-num1) == (num_ending2-num2):
     print("密度数据和波速数据在深度上已对齐，可以用于计算。")



###7.多余数据(深度、密度、波速)删除处理
del depth1[0:num1] #0：0(删前面0个数据)
del depth2[0:num2] #0：20
del depth1[num_ending1-num1:] #从下标2953开始删完后面的数据
del depth2[num_ending2-num2:] #从新下标2953开始删
if len(depth1) == len(depth2):
    print("深度数据已对齐，可以进行下一步操作。")

# 再对list1密度数据和list2波速数据进行对齐处理
del list1[0:num1]
del list1[num_ending1-num1:]
del list2[0:num2]
del list2[num_ending2-num2:]
if len(list1) == len(list2):
    print("密度数据和波速数据已对齐，可以进行下一步操作。")
print(len(list1))#3823



###8.删掉不合理的异常数据
# 除去密度中间部分不合理的数值
i = 0
while i < len(list1):
    if list1[i] > 10 or list1[i]<0:
        del list2[i]
        del depth2[i]
        del depth1[i]
        del list1[i]
        i = i-1
    i = i+1
print(len(list1))#1633
print(depth1)



# 除去速度中间部分不合理的数值
i = 0
while i < len(list2):
    if list2[i] > 10 or list2[i]<0:
        del list2[i]
        del depth2[i]
        del depth1[i]
        del list1[i]
        i = i-1
    i = i+1
print(len(list1))#3788
print(depth1)




###9.计算需要的物理量，反射系数、波阻抗
list = []
list_1 = []

# 计算反射系数
for i in range(0,len(depth1)-1):
    k1 = list1[i]*list2[i]
    k2 = list1[i+1]*list2[i+1]
    k = (k2-k1)/(k2+k1) #此处k即反射系数
    list.append(k)

# 计算波阻抗
for i in range(0,len(depth1)):
    I = list1[i]*list2[i]
    list_1.append(I)

# depth1.pop() #pop()函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值



###10.反射系数数据存储

with open("D:\\mlp-ledo-数据处理\\反射系数数据\\138-845A.txt","w")as file:
    file.writelines([str(line) +"\n"for line in list])
print("反射系数数据已保存。")


###11.作图：反射率直方图
sns.set() #设置绘图风格，seaborn.set()使用5种默认风格

name="138-845A"
name_1="94°35.448W,9°34.950'N"



ax2 = plt. gca()
ax2. set_title(str(name_1)+','+str(name)+',拟合t分布',fontproperties = songti)
ax2. set_xlabel("反射系数",fontproperties = songti)
ax2. set_ylabel("频率/组距",fontproperties = songti)
#ax2. legend([r"r",r"b"],fontsize=18,edgecolor="black",loc='lower right', frameon=True)
#plt.legend([r"a",r"b"],fontsize=18,edgecolor="black",loc='lower right', frameon=True)
sns. distplot(list,bins=100,kde=True,fit=t,kde_kws={"label":"138-845A-KDE"})
plt.legend()
#plt.legend(labels=["133-817D-fit-t"])
#kde=True表示绘制核密度估计图，bins用于控制条形的数量，fit控制拟合的参数分布图形
#norm_hist：若为True, 则直方图高度显示密度而非计数(含有kde图像中默认为True)
savefig('D:\\mlp-ledo-数据处理\\数据绘图\\138-845A\\'+name+'反射系数拟合t分布.png')
print("反射系数直方图已保存。")
plt.show()


# Get the fitted parameters used by sns返回拟合函数的参数
(df,loc,scale) = t.fit(list)
print("df={0},loc={1},scale={2}".format(df,loc,scale))
#scipy.stats.t的fit方法返回(df, loc, scale)



#绘制核密度估计曲线
ax2 = plt. gca()
ax2. set_title(str(name_1)+','+str(name)+',反射系数核密度估计曲线',fontproperties = songti)
ax2. set_xlabel("反射系数",fontproperties = songti)
sns.kdeplot(list,kernel='gau',bw='scott',shade=True)
savefig('D:\\mlp-ledo-数据处理\\数据绘图\\138-845A\\'+name+'反射系数核密度估计曲线.png')
plt.show()




