import numpy as np
import pandas as pd


def nmf(v, lambda1, lambda2, epislon, k, times):
    n, m = v.shape
    W = np.random.random((n, k))
    H = np.random.random((k, m))

    for time in range(times):
        W2 = W.copy()
        H2 = H.copy()
        VH = np.matmul(v, H2.T)
        WHH = np.matmul(np.matmul(W2, H2), H2.T)
        WV = np.matmul(W2.T, v)
        WWH = np.matmul(np.matmul(W2.T, W2), H2)

        for i in range(n):
            for j in range(k):
                W[i][j] = max((VH[i][j] * W2[i][j] - lambda1 * W2[i][j] ** 2) / (WHH[i][j] + epislon), 0)

        for p in range(k):
            for j in range(m):
                H[p][j] = max((WV[p][j] * H2[p][j] - lambda2 * H2[p][j] ** 2) / (WWH[p][j] + epislon), 0)

        if np.sum((W - W2) ** 2) < 0.001 and np.sum((H - H2) ** 2) < 0.001:
            break

    return W, H


