import numpy as np
import pandas as pd
import time
import math
from sklearn.model_selection import KFold

'''
使用公开数据集，math2015中的FrcSub
'''
def trainDINAModel(dataset, Q):
    # sg迭代的阈值
    threshold = 5

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
    # 学生*模式数 = 学生*  题目数     题目数*技能         技能*模式数
    while continueSG == True:
        # E步，求似然矩阵
        IL = np.zeros((ni, 2 ** k))
        # 技能模式的数量
        ### 算出所有模式组合的估计, 即对于不同的alphal，得到不同的L(X|alphal)
        for l in range(2 ** k):
            # P(Xij|alphal)
            lll = ((1 - sg[:, 0]) ** dataset * sg[:, 0] ** (1 - dataset)) ** r.T.A[l] * (sg[:, 1] ** dataset * (1 - sg[:, 1]) ** (1 - dataset)) ** (1 - r.T.A[l])
            # L(Xi|alphal)，这里的连乘是对于题目来说
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
    return sg, r, K

def predictDINA(n,Q,sg,r,K):
    startTime = time.time()
    print('预测开始')

    ni, nj = n.shape
    Qi, Qj = Q.shape
    # 预测的每个学生的技能向量
    IL = np.zeros((ni, 2 ** Qj))

    for l in range(2 ** Qj):
        # 学生的数量
        lll = ((1 - sg[:, 0]) ** n * sg[:, 0] ** (1 - n)) ** r.T.A[l] * (sg[:, 1] ** n * (
            1 - sg[:, 1]) ** (1 - n)) ** (1 - r.T.A[l])
        IL[:, l] = lll.prod(axis=1)
    # 只需要在上面的IL中，针对每一个学生，寻找他在所有l模式中，似然函数最大的那一项l
    a = IL.argmax(axis=1)

    i, j = n.shape
    precision = np.sum((r[:, a] == n.T) * 1) / (i * j)
    print('总共有' + str(ni) + '个人，a准确率为：', precision)

    skill_stat = n * Q
    skill_pred = K[:, a].T

    FN = np.sum((skill_pred == 0) & (skill_stat > 0) * 1)
    TP = np.sum((skill_pred == 1) & (skill_stat > 0) * 1)
    FP = np.sum((skill_pred == 1) & (skill_stat == 0) * 1)
    # 将预测为1的skill与stat的分数做mask，然后与总的stat的mask相除
    precision2 = TP/(TP+FP)
    score_precision2 = np.sum(skill_pred.A * skill_stat.A) / np.sum(skill_stat)

    # 计算召回，即预测为0但实际skill大于0的概率及分数
    recall2 = TP / (TP + FN)
    score_recall2 = np.sum(((skill_pred == 0) & (skill_stat > 0) * 1).A * skill_stat.A) / np.sum(skill_stat)
    print("precision:{} | {}".format(precision2,score_precision2))
    print("recall:{} | {}".format(recall2, score_recall2))

    print('预测消耗时间：' + str(int(time.time()) - int(startTime)) + '秒')
    print('-----------------------------------------------')

if __name__ == '__main__':
    # 数据读取
    raw = pd.read_csv('./math2015/FrcSub/data.txt', sep='\t', header=None).values
    Q = np.mat(pd.read_csv('./math2015/FrcSub/q.txt', sep='\t', header=None))

    n_splits = 5
    KF = KFold(n_splits=n_splits, shuffle=True)
    for train_index, test_index in KF.split(raw):
        X_train, X_test = raw[train_index], raw[test_index]
        sg,r, K = trainDINAModel(X_train,Q)
        predictDINA(X_test, Q, sg, r, K)


