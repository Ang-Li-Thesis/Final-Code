#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8
from __future__ import division, print_function
import numpy as np
from numpy.linalg import norm
import pandas as pd
from sklearn import svm, metrics, datasets
from scipy.special import gammaincinv
from collections import namedtuple
from sympy import fwht

# In[12]:


def fastfood_params(n, d):
    rng = np.random.RandomState(None)
    d0 = d
    l = int(np.ceil(np.log2(d)))
    d = 2**l
    k = int(np.ceil(n / d))
    n = d * k
    B = []
    G = []
    PI = []
    S = []
    for ii in range(k):
        B_ii = rng.choice([-1, 1], size=d)
        G_ii = rng.normal(size=d)
        PI_ii = rng.permutation(d)
        B.append(B_ii)
        G.append(G_ii)
        PI.append(PI_ii)
        p1 = rng.uniform(size=d)
        p2 = d / 2
#        print('p1 =',p1,'; p2 =',p2)
        T = gammaincinv(p2, p1)
#        print('T1 =',T)
        T = (T * 2) ** (1 / 2)
#        print('T2 =',T)
        s_i = T * norm(G, 'fro')**(-1)
#        print('s_i =', s_i)
        S_ii = s_i
        S.append(S_ii)
    S1 = np.zeros(n)
    for ii in range(k):
        S1[ii * d:(ii + 1) * d] = S[ii]
    FFPara = namedtuple('FFPara', 'B G PI S')
    return FFPara(B, G, PI, S1)


# In[6]:


def bit_reverse_traverse(a):
    n = a.shape[0]
    assert(not n & (n - 1))  # assert that n is a power of 2
    if n == 1:
        yield a[0]
    else:
        even_index = np.arange(int(n / 2)) * 2
        odd_index = np.arange(int(n / 2)) * 2 + 1
        for even in bit_reverse_traverse(a[even_index]):
            yield even
        for odd in bit_reverse_traverse(a[odd_index]):
            yield odd


# In[7]:


def get_bit_reversed_list(l):
    n = len(l)
    indexs = np.arange(n)
    b = []
    for i in bit_reverse_traverse(indexs):
        b.append(l[i])
    return b


# In[8]:


def FWHT(X):
#     print('before get_bit_reversed_list')
#     print(X.shape)
    x = get_bit_reversed_list(X)
#     print('afyer get_bit_reversed_list')
#     print(len(x))
    x = np.array(x)
    N = len(X)

    for i in range(0, N, 2):
        x[i] = x[i] + x[i + 1]
        x[i + 1] = x[i] - 2 * x[i + 1]

    L = 1
    y = np.zeros_like(x)
    for n in range(2, int(np.log2(N)) + 1):
        M = 2**L
        J = 0
        K = 0
        while(K < N):
            for j in range(J, J + M, 2):
                y[K] = x[j] + x[j + M]
                y[K + 1] = x[j] - x[j + M]
                y[K + 2] = x[j + 1] + x[j + 1 + M]
                y[K + 3] = x[j + 1] - x[j + 1 + M]
                K = K + 4
            J = J + 2 * M
        x = y.copy()
        L = L + 1

    y = x / float(N)

    return y


# In[18]:


def fastfood_forkernel(X, n):
    m, d0 = X.shape

    para = fastfood_params(n, d0)
    X = np.transpose(X)
    l = int(np.ceil(np.log2(d0)))
    d = 2**l
    gamma = 1 / (d0 * X.var())
#     gamma = 1/d
    sgm = np.sqrt(gamma)
    if d == d0:
        XX = X
    else:
        XX = np.zeros((d, m))
        XX[0:d0, :] = X
        
    k = len(para.B)
    n = d * k
    tht = np.zeros((n, m))
    for ii in range(k):
        B = para.B[ii]
        G = para.G[ii]
        PI = para.PI[ii]
        XX = np.dot(np.diag(B), XX)
        T = FWHT(XX)
#         T = fwht(XX)
#         T = np.array(T)
        T = T[PI, :]
        T = np.dot(np.diag(G * d), T)
#         T = fwht(T)
#         T = np.array(T)
        T = FWHT(T)
        idx1 = ii * d
        idx2 = (ii + 1) * d
        tht[idx1:idx2, :] = T
    S = para.S
    tht = (S * np.sqrt(d) * tht.T).T
    T = tht / sgm
#     T = tht / (d * sgm)
    b = np.random.uniform(-np.pi, np.pi, len(T))
    phi = np.cos(np.transpose(T) + b) 
    phi = np.sqrt(2/n) * phi
    return phi


# In[19]:


# def ff_kernel(para, sgm, x1, x2):
#     X = np.hstack((x1, x2))
#     phi, tht = fastfood_forkernel(X, para, sgm)
#     K_appro = np.dot(phi[0].T, phi[1])
#     return K_appro


# In[ ]:


# para = fastfood_params(n, d)
# print('Fastfood params:',params)


# In[27]:


# def compute_kernel_matrix(X, phi):
#     X = np.transpose(X)
#     d0, m = X.shape
# #     X = np.transpose(X)
#     l = int(np.ceil(np.log2(d0)))
#     d = 2**l
#     if d == d0:
#         XX = X
#     else:
#         XX = np.zeros((d, m))
#         XX[0:d0, :] = X
# #     print('XX.shape:',XX.shape)
#     K = np.zeros((m, m), dtype=np.float64)
#     for i in range(m):
#         for j in range(m):
#             phi_i = phi[i]
#             phi_j = phi[j]
#             K_ij = np.dot(phi_i, phi_j)
#             K[i, j] = K_ij
#     return K


# In[28]:


# X1T = X1.T
# print('X1.shape:',X1.shape)
# phi1, tht1 = fastfood_forkernel(X1T, para, sgm)
# # print(phi1[1:3,:])
# # print('phi1.shape:',phi1.shape)
# # phi1 = pd.read_csv(
# #     '/Users/kellanfluette/dev/fastfood/samples/matlab_fastfood/' +
# #     'Fastfood/phi1.csv', header=None)
# K_appro = compute_kernel_matrix(X1T, phi1)
# # print('K_appro =',K_appro)
# # print('K_appro.shape:',K_appro.shape)

# C = 1.0
# gamma = 0.001


# # In[ ]:


# # we create an instance of SVM and fit out data.
# # my_kernel = lambda x1,x2: ff_kernel(para,sgm,x1,x2)
# clf = svm.SVC(kernel='precomputed', gamma=gamma, C=C, random_state=rng)
# clf.fit(K_appro, y)
# y_pred = clf.predict(K_appro)
# print('ff-kSVM metrics:\n', metrics.classification_report(y, y_pred))

# clf2 = svm.SVC(kernel='rbf', C=C, gamma=gamma, random_state=rng)
# # print('X1.shape:',X1.shape,', y.shape:',y.shape)
# clf2.fit(X1, y)
# y_pred = clf2.predict(X1)
# print('kSVM(rbf) metrics:\n', metrics.classification_report(y, y_pred))

