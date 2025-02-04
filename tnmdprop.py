#Tensor network propagation

import numpy as np
from scipy.linalg import expm
import time
import sys
sys.path.insert(1, '/N/project/MQCD/Multi_Dim_Regularization/codes')
import modprop
import modified_round
import pickle
import logging

np.seterr(all='warn')

logging.basicConfig(filename="newfile.log", format='%(asctime)s %(message)s',filemode='w')
logger = logging.getLogger()
logger.info("Getting Inputs")

Time1 = time.time()
Ndim = int(sys.argv[1])
arr = sys.argv[2].split(',')
d = np.array([int(i) for i in arr])
kinetic = {}
for i in range(Ndim):
    kinetic['%s'%i] = np.loadtxt('/N/project/MQCD/Multi_Dim_Regularization/data/K49.txt') 
    #kinetic['%s'%i] = np.zeros((d[i],d[i])) 	
    #kinetic['%s'%i] = np.loadtxt('k%s.txt'%i)

t = float(sys.argv[3])#timestep
hbar = float(sys.argv[4])
thres1 = float(sys.argv[5])#for vectors
#steps = int(sys.argv[8])
T = float(sys.argv[6])#Total prop time
Temp = float(sys.argv[7])
wf = int(sys.argv[8])
regulate= int(sys.argv[9])
path1 = sys.argv[10]
steps = int(T/t)

DEBUG = True
DEBUG1 = False
TNtype = 2

#*************************************************************
if DEBUG:
    print('Inputs')
    print('Ndim, d =', Ndim, d)
    print('t, hbar, thres1, T, Temp, steps=', t, hbar, thres1, T, Temp, steps)
    print('wftype, TNtype=', wf, TNtype)
    print('regulate', regulate)
#************************************************************

expk = modprop.kinpropagator(kinetic, t, hbar)

UP = np.loadtxt('%s/regulated_bond_dimension%s.txt'%(path1,t), dtype=complex)

with open('%s/Ucx%s.pkl'%(path1,t), 'rb') as f:
    Ucx = pickle.load(f) 

#with open('Uround10.0.pkl', 'rb') as f: 
 #    Ucx = pickle.load(f) 

if DEBUG:
    for i in range(Ndim):
        print('Ucx.shape =', Ucx['cx%s'%i].shape)
#wavefunction
'''
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
#X = modprop.Xd(ini_x, fin_x, d, sig, mu, Ndim)
#X0 = modprop.ini_Xd(X)

mu = np.full(Ndim, 0.5)
sig = np.full(Ndim, 0.12)
X = modprop.md_gauss_fn(Ndim, d, mu, sig)
X0 = modprop.ini_Xd(X)

print('steps =',steps)
print('Before propagation')
#cx, Qp = modprop.TNMDprop(Ures, X, Ndim, d, P, expk, thres, steps, TNtype)
cx, Qp = modprop.TNMDprop(Ucx, X, Ndim, d, UP, expk, thres1, steps, TNtype, regulate, DEBUG, DEBUG1)

if DEBUG:
    print('Qp =', Qp)

Qp1 = np.zeros((steps+1,Ndim-1))
for i in range(steps+1):
    for j in range(Ndim-1):
        Qp1[i,j] = Qp['Qp%s'%i][j] 

np.savetxt('sqr%s.txt'%t, Qp1)

with open('cx%s.pkl'%t, 'wb') as f:
    pickle.dump(cx, f)

Time2 = time.time()
Time = Time2-Time1

print('Time taken =', Time)
print('Run Over Yay tnmdprop')
