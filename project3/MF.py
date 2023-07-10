import numpy as np
import pandas as pd
import time
from scipy.sparse import find


# 将矩阵R分解成P,Q
def matrix_factorization(R, P, Q, K, steps, alpha=0.05, Lambda=0.002):
    sum_st = 0  # 总时长

    E = np.zeros((201, 1421))  # Error
    P2 = P.copy()
    Q2 = Q.copy()
    index = np.array(find(R))  # 非0元素的索引以及数值
    for step in range(steps):
        # 每次迭代开始的时间
        st = time.time()
        cnt = 0
        e_new = 0
        for m in range(len(index[0])):
            u = int(index[0][m])
            i = int(index[1][m])
            val = index[2][m]
            eui = val - np.dot(P[u, :], Q[:, i])
            E[u][i] = eui

        # 梯度向量
        g1 = np.zeros((200, K))
        g2 = np.zeros((K, 1421))
        for m in range(len(index[0])):
            u = int(index[0][m])
            i = int(index[1][m])
            for t in range(K):
                g1[u][t] -= E[u][i] * Q2[t][i]
                g2[t][i] -= E[u][i] * P2[u][t]

        # 正则化
        for u in range(201):
            for j in range(K):
                g1[u][j] += Lambda * P2[u][j]
        for j in range(K):
            for i in range(1421):
                g2[j][i] += Lambda * Q2[j][i]

        for m in range(len(index[0])):
            u = int(index[0][m])
            i = int(index[1][m])
            for j in range(K):
                P[u][j] -= alpha * g1[u][j]
                Q[j][i] -= alpha * g2[j][i]

        if np.sum((P - P2) ** 2) < 0.000001 and np.sum((Q - Q2) ** 2) < 0.000001:
            break
        et = time.time()
        e = np.sum((P - P2) ** 2)

    print('---------Summary----------\n',
          'Type of jump out:',flag,'\n',
          'Total steps:',step + 1,'\n',
          'Total time:',sum_st,'\n',
          'Average time:',sum_st/(step+1.0),'\n',
          "The e is:", e, '\n',
          'Total RMSE: ', pow(e, 0.5))
    return P, Q


# 训练
def train(K, steps):
    R = pd.read_csv("/project3/dataset/rating_mat.txt", delimiter=',', dtype=int, header=None)
    M = 201
    N = 1421
    # 用户矩阵初始化
    P = np.random.normal(loc=0, scale=0.01, size=(M, K))
    # 项目矩阵初始化
    Q = np.random.normal(loc=0, scale=0.01, size=(K, N))
    P, Q = matrix_factorization(R, P, Q, K, steps)

    # 将分解后得到的P，Q保存到本地
    np.savetxt("D:/algorithm_project/project3/dataset/userMatrix.txt", P, fmt="%.6f", delimiter=',', newline='\n')
    np.savetxt("D:/algorithm_project/project3/dataset/movieMatrix.txt", Q, fmt="%.6f", delimiter=',', newline='\n')
    print("train complete!")


# 测试
def test(dformat):
    # 读取矩阵
    P = np.loadtxt("D:/algorithm_project/project3/dataset/userMatrix.txt", delimiter=',', dtype=float)
    Q = np.loadtxt("D:/algorithm_project/project3/dataset/movieMatrix.txt", delimiter=',', dtype=float)
    test = pd.read_csv("/project3/dataset/test.txt", sep=',', header=None)
    test = np.array(test)
    rmse = 0
    mae = 0
    cnt = np.size(test, 0)

    # 遍历，计算误差
    for item in range(cnt):
        i = test[item][0]
        j = test[item][1]
        score = test[item][2]
        predict = np.dot(P[i, :], Q[:, j])
        rmse += pow(predict - score, 2)
        mae += abs(predict - score)

    rmse = pow((rmse / cnt), 0.5)
    mae = mae / cnt
    print(rmse, mae)


# 输出前k名推荐
def recommend(user, k, P, Q):
    dic = {}  # key为电影编号，value为评分
    for j in range(len(Q[0])):
        dic[j] = np.dot(P[user, :], Q[:, j])

    res = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}
    for i, (movie, score) in enumerate(res.items()):
        if i in range(k):
            print("movie: ", movie, " score: ", score)


rtnames = ['user', 'movie', 'score', 'time']
train(18, 200)
test(rtnames)
