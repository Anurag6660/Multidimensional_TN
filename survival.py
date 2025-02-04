#Survival Probabilities

import numpy as np
from scipy.linalg import expm
import time
import sys
sys.path.insert(1, '/N/project/MQCD/Multi_Dim_Regularization/codes')
import modprop
import pickle

np.seterr(all='warn')

Time1 = time.time()
Ndim = int(sys.argv[1])
arr = sys.argv[2].split(',')
d = np.array([int(i) for i in arr])
t = float(sys.argv[3])#timestep
T = float(sys.argv[4])#Total prop time
steps = int(T/t)

DEBUG = True
DEBUG1 = False


#*************************************************************
if DEBUG:
    print('Inputs')
    print('Ndim, d =', Ndim, d)
    print('t, T, steps=', t, T, steps)
#************************************************************

with open('cx%s.pkl'%t, 'rb') as f:
    cx = pickle.load(f) 

Qp1 = np.loadtxt('sqr%s.txt'%t)

Qp = {}

for i in range(steps+1):
    Qp['Qp%s'%i] = np.zeros(Ndim-1, dtype=int)

for i in range(steps+1):
    for j in range(Ndim-1):
        Qp['Qp%s'%i][j] = Qp1[i,j]

comb0, Mcomb0 = modprop.recombine3D(cx[0], Ndim, d, Qp['Qp%s'%0])
sur_cx = np.zeros(steps, dtype=complex)
for i in range(steps):
    combx, Mcombx = modprop.recombine3D(cx[i], Ndim, d, Qp['Qp%s'%i])
    print('Mcombx.shape', Mcombx.shape)
    print('Mcomb0.shape', Mcomb0.shape)
    print('Mcomb0T.shape', (Mcomb0.conj().T).shape)
    print('sur_cx', (Mcomb0.conj().T)@Mcombx)
    sur_cx[i] = (Mcomb0.conj().T)@Mcombx

np.savetxt('surcx%s.txt'%t, sur_cx)
Time2 = time.time()
print('Time taken', Time2-Time1)
print('Survival Prob Run Over Yay!')

