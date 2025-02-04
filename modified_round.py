import numpy as np
from scipy.linalg import svd

def SVR(se, thres):
    count=0
    for i in range(len(se)):
        if (se[i]/se[0])>thres:
            count = count + 1
            P = count
    return P

def Us(Ue, se, P):
     U1 = Ue[:,0:P]*se[0:P]
     return U1

def U1W1(c, thres):
    Ue, se, We = np.linalg.svd(c, full_matrices= False)
    P = SVR(se, thres)
    U1 = Us(Ue, se, P)
    # Fix Transponse for performance.
    W1 = We[0:P,:]
    #W1T = We.T[:,0:P]
    #W1 = W1T.T
    return U1, W1, se, P

def round(vx, Ndim, d, thres):
    
    Qp = np.zeros(Ndim-1, dtype=int)
    core0 = vx['cx%s'%0]
    nrm = np.zeros(d)
    for i in range(Ndim-1):
        core0 = core0.reshape(-1, vx['cx%s'%i].shape[-1], order = 'F')
        core0, ru = np.linalg.qr(core0, mode='reduced')
        #nrm[i] = np.linalg.norm(ru, 'fro')
        #if nrm[i] != 0:
         #   ru /= nrm[i]
        core1 = vx['cx%s'%(i+1)]
        core1 = core1.reshape(ru.shape[1], -1, order='F')
        core1 = ru @ core1
        vx['cx%s'%i] = core0
        core0 = core1

    #vx['cx%s'%(Ndim-1)] = ru@vx['cx%s'%(Ndim-1)].T
    #for i in range(1, Ndim-1):
    #    vx['cx%s'%i] = vx['cx%s'%i].reshape(vx['cx%s'%(i-1)].shape[-1],d[i],-1, order='F')
   
    #for i in range(Ndim-1):
     #   Qp[i] = vx['cx%s'%i].shape[-1]

    vx['cx%s'%(Ndim-1)] = (ru@vx['cx%s'%(Ndim-1)].T).T
    #print('round', vx['cx%s'%0].shape, vx['cx%s'%1].shape, vx['cx%s'%2].shape, vx['cx%s'%3].shape)
    
    core0 =  vx['cx%s'%(Ndim-1)].T
    for i in range(Ndim-1, 0, -1):
        core0 = core0.reshape(vx['cx%s'%(i-1)].shape[-1],-1, order='F')
        u, w, _, Qp[i-1] = U1W1(core0, thres)
        core1 = vx['cx%s'%(i-1)]@u
        vx['cx%s'%i] = w
        core0 = core1
        #print('i, vx[(i-1)].shape, u.shape, w.shape', i, vx['cx%s'%(i-1)].shape, u.shape, w.shape)

    for i in range(1, Ndim-1):
        vx['cx%s'%i] = vx['cx%s'%i].reshape(vx['cx%s'%i].shape[0],d[i],-1, order='F')
    
    vx['cx%s'%0] = core0
    vx['cx%s'%(Ndim-1)] = vx['cx%s'%(Ndim-1)].T
    return vx, Qp


