import os
import time

import numpy as np
from PIL import Image
import math


# 幂法求特征值
def mifa(cov):
    a = cov
    n = a.shape[0]
    eigval = []
    eigvec = []
    for k in range(n):
        v = []
        for i in range(n):
            v.append(1)
        v = np.array(v)
        l_now = np.norm(v)
        l_prev = 0
        u = (1 / l_now) * v
        while abs(l_now - l_prev) > 0.001:
            v = np.dot(a, u)
            l_prev = l_now
            l_now = math.sqrt(v[0] * v[0] + v[1] * v[1])
            u = (1 / l_now) * v

        eigval.append(l_now)
        vec = u.reshape(-1, 1)
        eigvec.append(vec)
        a = a - np.matmul(u.T, u)

    eigval = np.array(eigval)
    eigvec = np.array(eigvec)
    return eigval, eigvec


# Gram-Schmidt
def gram_schmidt(A):
    Q = np.zeros_like(A)
    cnt = 0
    for a in A.T:
        u = np.copy(a)
        for i in range(0, cnt):
            u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i])
        e = u / np.linalg.norm(u)
        Q[:, cnt] = e
        cnt += 1
    R = np.dot(Q.T, A)
    return Q, R


def qr_gs(a):
    a_now = a
    l = a.shape[0]
    vec = np.eye(l)
    for i in range(10):
        q, r = gram_schmidt(a_now)
        a_now = np.matmul(r, q)
        vec = np.matmul(vec, q)
    val = []
    for i in range(l):
        val.append(a_now[i, i])
    val = np.array(val)
    return val, vec


# Householder
def householder(A):
    (r, c) = np.shape(A)
    Q = np.identity(r)
    R = np.copy(A)
    for cnt in range(r - 1):
        x = R[cnt:, cnt]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        Q_cnt = np.identity(r)
        Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, v)
        R = np.dot(Q_cnt, R)
        Q = np.dot(Q, Q_cnt)
    return Q, R


def qr_hh(a):
    a_now = a
    l = a.shape[0]
    vec = np.eye(l)
    for i in range(10):
        q, r = householder(a_now)
        a_now = np.matmul(r, q)
        vec = np.matmul(vec, q)
    val = []
    for i in range(l):
        val.append(a_now[i, i])
    val = np.array(val)
    return val, vec


# Schwarz-Rutishauser
def schwarz(A, type=complex):
    A = np.array(A, dtype=type)
    (m,n) = np.shape(A)
    Q = np.array(A, dtype=type)
    R = np.zeros((n, n), dtype=type)
    for k in range(n):
        for i in range(k):
            R[i, k] = np.transpose(Q[:, i]).dot(Q[:, k]);
            Q[:, k] = Q[:, k] - R[i, k] * Q[:, i];
        R[k, k] = np.linalg.norm(Q[:, k]); Q[:, k] = Q[:, k] / R[k, k];
    return -Q, -R


def qr_sr(a):
    a_now = a
    l = a.shape[0]
    vec = np.eye(l)
    for i in range(10):
        q, r = schwarz(a_now)
        a_now = np.matmul(r, q)
        vec = np.matmul(vec, q)
    val = []
    for i in range(l):
        val.append(a_now[i, i])
    val = np.array(val)
    return val, vec


# PCA的主循环部分
runtime = []  # 运行时间
compress = []  # 压缩率
mse = []  # 重构误差
for i in range(100):
    # 读入图片
    path = 'D:/algorithm_project/project2/Images/airplane/airplane'
    if i <= 9:
        path = path + '0' + str(i) + '.tif'
    else:
        path = path + str(i) + '.tif'
    img = Image.open(path)
    img = np.array(img.convert('RGB'))

    # 平铺成一个二维数组
    img1 = np.hstack((img[:, :, 0], img[:, :, 1], img[:, :, 2]))
    # 当前图片原始大小,KB
    size_before = (os.path.getsize(path)) / 1024
    # 开始计时
    t1 = time.perf_counter()

    # 标准化
    mean = np.mean(img1, axis = 0).reshape(1, -1)
    std = np.std(img1, axis = 0).reshape(1, -1)
    img2 = (img1 - mean) / std

    # 计算协方差矩阵
    cov = np.matmul(np.transpose(img2), img2)

    # 计算特征值、特征向量
    eigval, eigvec = mifa(cov)  # 这里可替换为其他分解方法，调用不同函数
    # eigval, eigvec = qr_gs(cov)
    # eigval, eigvec = qr_hh(cov)
    # eigval, eigvec = qr_sr(cov)
    seq = np.flip(np.argsort(eigval))
    eigvec = eigvec[:, seq]
    eigval = eigval[seq]

    # 计算保留特征值个数k
    total = sum(eigval)
    s = 0
    k = 0
    for item in eigval:
        s += item
        k += 1
        if s >= 0.95 * total:
            break

    # 降维、重构
    score = np.matmul(img2, eigvec)
    img3 = np.matmul(score[:, :k], eigvec.T[:k, :])
    img3 = img3 * std + mean
    img3 = img3.astype('uint8')

    # 重新变为3通道输出
    img3_channel = np.hsplit(img3, 3)
    img4 = np.zeros_like(img)
    for j in range(3):
        img4[:, :, j] = img3_channel[j]
    res = Image.fromarray(img4)
    final_path = 'D:/algorithm_project/project2/Images/result1/' + str(i) + '.tif'
    res.save(final_path)

    # 停止计时
    t2 = time.perf_counter()
    t = t2 - t1  # 输出单位为秒
    runtime.append(t)
    # 计算压缩率
    size_after = (os.path.getsize(final_path)) / 1024
    rate = (size_before - size_after) / size_before
    compress.append(rate)

    # 计算重构误差
    mse1 = np.sum((img3 - img1) ** 2) / (256 * 768)
    mse.append(mse1)

print(sum(runtime) / 100)
print(sum(compress) / 100)
print(sum(mse) / 100)
