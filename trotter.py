#Trotter propagation


import numpy as np
from scipy.linalg import expm
import time
import sys
sys.path.insert(1, '/N/project/MQCD/Multi_Dim_Regularization/codes')
import modprop
import pickle

np.seterr(all='warn')

Time1 = time.time()
A = np.loadtxt(sys.argv[1])#Potential
Ndim = int(sys.argv[2])
arr = sys.argv[3].split(',')
d = np.array([int(i) for i in arr])
kinetic = {}
for i in range(Ndim):
    kinetic['%s'%i] = np.loadtxt('/N/project/MQCD/Multi_Dim_Regularization/data/K49.txt') 
    #kinetic['%s'%i] = np.loadtxt('K49.txt') 

t = float(sys.argv[4])#timestep
hbar = float(sys.argv[5])
#steps = int(sys.argv[8])
T = float(sys.argv[6])#Total prop time
wf = int(sys.argv[7])
V = A-A.min()
steps = int(T/t)

DEBUG = True
DEBUG1 = False


#*************************************************************
if DEBUG:
    print('Inputs')
    print('Ndim, d =', Ndim, d)
    print('t, hbar, T, steps=', t, hbar, T, steps)
    print('V.shape =', V.shape)
    print('Shift in potential =', A.min())
#************************************************************

#Reshaping potential into x1 by x2x3...xn dimension
A1 = np.reshape(V, (d[0], np.prod(d[1:Ndim])), order = 'F')

#Calculate potential and kinetic propagators
Den = 2.0
c = np.exp(-1j*A1*t/(Den*hbar))
expk = modprop.kinpropagator(kinetic, t, hbar)

'''
#wavefunction
if (Ndim==2):
    ini_x = np.array([0.011, -40])
    fin_x = np.array([1.089, 40])
    sig = np.array([0.252, 16.99])
    mu = np.array([0.55, 0])

elif (Ndim==3):
    ini_x = np.array([0,0,0])
    fin_x = np.array([0.8,0.8,0.8])
    sig = np.array([0.12,0.12,0.12])
    mu = np.array([0.4,0.4,0.4])

elif (Ndim==4):
    ini_x = np.array([0,0,0,0])
    fin_x = np.array([0.8,0.8,0.8,0.8])
    sig = np.array([0.12,0.12,0.12,0.12])
    mu = np.array([0.4,0.4,0.4,0.4])
'''

mu = np.full(Ndim, 0.5) 
sig = np.full(Ndim, 0.12) 
X = modprop.md_gauss_fn(Ndim, d, mu, sig)
X0 = modprop.ini_Xd(X)

with open('cx%s.pkl'%t, 'rb') as f:
    cx = pickle.load(f) 
'''
Qp1 = np.loadtxt('sqr%s.txt'%t)

Qp = {}

for i in range(steps+1):
    Qp['Qp%s'%i] = np.zeros(Ndim-1, dtype=int)

for i in range(steps+1):
    for j in range(Ndim-1):
        Qp['Qp%s'%i][j] = Qp1[i,j]
'''
#Trotter     
X2 = np.reshape(X0, (d[0],-1))
#trot3d = modprop.trotprop(c, expk, X2, Ndim, d, steps)
Errtrot = np.zeros((steps, 2))
for i in range(steps):
    #combx, Mcombx = modprop.recombine3D(cx[i], Ndim, d, Qp['Qp%s'%i])
    combx, Mcombx = modprop.recombine3D(cx[i], Ndim)
    print('normcombx =', np.linalg.norm(combx))
    _, CVX2 = modprop.trotprop(c, expk, X2, Ndim, d, 1)
    X2 = CVX2
    errortrot = np.linalg.norm(CVX2 - combx)
    normtrot = np.linalg.norm(CVX2)
    print('Errortrot i', i)
    Errtrot[i,0] = i
    Errtrot[i,1] = errortrot
    print('normtrot =', normtrot)
np.savetxt('Errtrotter%s.txt'%t, Errtrot)
Time2 = time.time()
print('Time taken', Time2-Time1)
print('Trotter Run Over Yay!')

