#Calculating TN of potential propagator

import numpy as np
import time
import sys
sys.path.insert(1, '/N/project/MQCD/Multi_Dim_Regularization/codes')
import modprop
import modified_round
import pickle
import logging

np.seterr(all='warn')
#logger = logging.getLogger()
#=================================================================
# Inputs
# A = Potential (in Hartree units)
# Ndim = No. of dimension
# d = no. of grid points along each dimension
# t = timestep
# hbar = 1.0
# thres = threshold for potential propagator
# thres1 = threshold for propagated wavefunction used while doing QRSVD
#==================================================================
logging.info("TNpotprop code begins")
A = np.loadtxt(sys.argv[1], dtype=np.float64)#Potential
Ndim = int(sys.argv[2])
arr = sys.argv[3].split(',')
d = np.array([int(i) for i in arr])
t = float(sys.argv[4])#timestep
hbar = float(sys.argv[5])
thres = float(sys.argv[6])#for potential and prop
V = A-A.min()
DEBUG = True
DEBUG1 = True
regulate = int(sys.argv[7])

logging.info("Got the inputs")
#*************************************************************
if DEBUG:
    print('Inputs')
    print('A.shape =', A.shape)
    print('Ndim, d =', Ndim, d)
    print('t, hbar, thres=', t, hbar, thres)
    print('V.shape =', V.shape)
    print('Shift in potential =', A.min())
    print('DEBUG, DEBUG1 =', DEBUG, DEBUG1)
    print('regulate =', regulate)
#************************************************************
logging.info("Reshaping the potential")
#Reshaping potential into x1 by x2x3...xn dimension
A1 = np.reshape(V, (d[0], np.prod(d[1:Ndim])), order = 'F')

np.savetxt('Potential_%sD_reshaped_tushar.txt'%Ndim, A1)

logging.info("Performing TN of potential propagator")
#TN of potential propagator
Ures, s, P = modprop.AUWSP(A1, thres, Ndim, d)


logging.info("Saving the TN form of potential propagator naming Ures and singular values")
with open('Ures%s.pkl'%t, 'wb') as f:
    pickle.dump(Ures, f)

with open('singU_v%s.pkl'%t, 'wb') as f:
    pickle.dump(s, f)

size = 0
for i in range(len(s)):
    size1 = len(s['%s'%i])
    if size1>size:
        size=size1

for i in range(len(s)):
    s['%s'%i] = np.pad(s['%s'%i], (0,size-len(s['%s'%i])), 'constant', constant_values =0)

se = np.zeros((size,Ndim-1), dtype=np.float64)
for i in range(Ndim-1):
    se[:,i] = s['%s'%i]

np.savetxt('se%s.txt'%t, se)
    


'''
logging.info("If regulate == 1, performing regularizarion algo on TN of potential propagator, if regulate == 2, performing rounding algorithm on potential propagator")

if (regulate == 1):
    Ucx, PU, sR = modprop.MDQRSVD(Ures, Ndim, d, thres)

elif (regulate == 2):
    Ucx, PU = modified_round.round(Ures, Ndim, d, thres)
    
logging.info("Saving the regulated TN form of potential propgator naming Ucx")
with open('Ucx%s.pkl'%t, 'wb') as f:
     pickle.dump(Ucx, f) 

np.savetxt('bond_dimension%s.txt'%t, P)
np.savetxt('regulated_bond_dimension%s.txt'%t, PU)
#***************************************************************
if DEBUG:
    for i in range(len(Ures)):
        print('Ures.shape = ',Ures['cx%s'%i].shape)
        print('Ucx.shape = ',Ucx['cx%s'%i].shape)
'''
#check for AUWSP
if DEBUG1:
  # prodU, MdU = modprop.recombine3D(Ures, Ndim, d, P)
   prodU, MdU = modprop.recombine3D(Ures, Ndim, d, P)
   normV = np.linalg.norm(A1-prodU)
   print('normV = ',normV)

print('TNpotprop Run Over Yay')
