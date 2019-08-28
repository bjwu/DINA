import numpy as np
import pandas as pd
import time
import math
from multiprocessing import Pool

'''
使用公开数据集，math2015中的FrcSub
'''

# sg迭代的阈值
threshold = 5

# 数据读取
dataset = pd.read_csv('./math2015/FrcSub/data.txt', sep='\t', header=None).values
Q = np.mat(pd.read_csv('./math2015/FrcSub/q.txt', sep='\t', header=None))


startTime = time.time()
print('开始训练DINA模型')

# ni: 被试人数
# nj: 题目个数
# Qi: 题目个数
# k: skill维度
ni, nj = dataset.shape
Qi, k = Q.shape

# 计算每道题目的s失误率和g猜测率
sg = np.zeros((nj, 2))

#构造K矩阵，表示k个技能可以组成的技能模式矩阵
#每一列代表一个k位二进制数，表示一个拥有特定skills组合的模式
K = np.mat(np.zeros((k, 2 ** k), dtype=int))
for j in range(2 ** k):
    l = list(bin(j).replace('0b', ''))
    for i in range(len(l)):
        K[k - len(l) + i, j] = l[i]


# 评判标准
std = np.sum(Q, axis=1)
#r矩阵表示理论上j这道题目对于l这个模式能否做对，相对于公式中的eta矩阵
r = (Q * K == std) * 1


# 初始化每道题目的s失误率和g猜测率，
# sg[i][0]表示第i道题目的s失误率，sg[i][1]表示第i道题目的g猜测率
for i in range(nj):
    sg[i][0] = 0.2
    sg[i][1] = 0.2

continueSG = True
kk =1
lastLX = 1
# 计算s和g迭代的次数
# 学生*模式数 = 学生*  题目数         题目数*技能         技能*模式数
while continueSG == True:
    # E步，求似然矩阵
    IL = np.zeros((ni, 2 ** k))
    # 技能模式的数量
    print('single process')
    ### 算出所有模式组合的估计, 即对于不同的alphal，得到不同的L(xi|alpha)
    for l in range(2 ** k):
        lll = ((1 - sg[:, 0]) ** dataset * sg[:, 0] ** (1 - dataset)) ** r.T.A[l] * (sg[:, 1] ** dataset * (1 - sg[:, 1]) ** (1 - dataset)) ** (1 - r.T.A[l])
        # L(xi|alpha)，这里的连乘是对于题目来说
        IL[:,l] = lll.prod(axis=1)
    sumIL = IL.sum(axis=1)
    LX = np.sum([i for i in map(math.log2, sumIL)])
    print('LX: ', LX)
    # 所以这里假设了P(alphal)的先验概率都相等
    IL = (IL.T / sumIL).T
    #IR中的 0 1 2 3  分别表示 IO RO I1 R1
    IR = np.zeros((4,nj))
    n1 = np.ones(dataset.shape)
    for l in range(2 ** k):
        IR[0] += np.sum(((1-r.A[:,l])* n1).T*IL[:,l],axis=1)
        IR[1] += np.sum(((1-r.A[:,l])* dataset).T*IL[:,l],axis=1)
        IR[2] += np.sum((r.A[:,l]* n1).T*IL[:,l],axis=1)
        IR[3] += np.sum((r.A[:,l]* dataset).T*IL[:,l],axis=1)
    #针对每一道题目，根据I0,R0,I1,R1，来更新s和g，更新后的sg，又重新计算似然函数矩阵IL
    # if (abs(IR[1] / IR[0] - sg[:,1])<threshold).any() and (abs((IR[2]-IR[3]) / IR[2] -sg[:,0])<threshold).any():
    if abs(LX-lastLX)<threshold:
        continueSG = False

    lastLX = LX
    sg[:,1] = IR[1] / IR[0]
    sg[:,0] = (IR[2]-IR[3]) / IR[2]
    print(str(kk ) +"次迭代，"+str(ni)+"个学生，"+str(nj)+"道题目的失误率和猜测率")
    kk +=1
    a=0
endTime = time.time()
print('DINA模型训练消耗时间：'+str(int(endTime-startTime))+'秒')

