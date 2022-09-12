import numpy as np
import torch
from numba import jit
from torch.autograd import Function

@jit(nopython = True)
def compute_softdtw(D, gamma):
  N = D.shape[0]
  M = D.shape[1]
  R = np.zeros((N + 2, M + 2)) + 1e8
  R[0, 0] = 0
  for j in range(1, M + 1):
    for i in range(1, N + 1):
      r0 = -R[i - 1, j - 1] / gamma
      r1 = -R[i - 1, j] / gamma
      r2 = -R[i, j - 1] / gamma
      rmax = max(max(r0, r1), r2)
      rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
      softmin = - gamma * (np.log(rsum) + rmax)
      R[i, j] = D[i - 1, j - 1] + softmin
  return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
  N = D_.shape[0]
  M = D_.shape[1]
  D = np.zeros((N + 2, M + 2))
  E = np.zeros((N + 2, M + 2))
  D[1:N + 1, 1:M + 1] = D_
  E[-1, -1] = 1
  R[:, -1] = -1e8
  R[-1, :] = -1e8
  R[-1, -1] = R[-2, -2]
  for j in range(M, 0, -1):
    for i in range(N, 0, -1):
      a0 = (R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma
      b0 = (R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma
      c0 = (R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma
      a = np.exp(a0)
      b = np.exp(b0)
      c = np.exp(c0)
      E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
  return E[1:N + 1, 1:M + 1]
 

class SoftDTW(Function):
    @staticmethod
    def forward(ctx, D, gamma = 1.0): # D.shape: [batch_size, N , N]
      device = D.device
      batch_size,N,N = D.shape
      gamma = torch.FloatTensor([gamma]).to(device)
      D_ = D.detach().cpu().numpy()
      g_ = gamma.item()

      total_loss = 0
      R = torch.zeros((batch_size, N+2 ,N+2)).to(device)   
      for k in range(0, batch_size): # loop over all D in the batch    
        Rk = torch.FloatTensor(compute_softdtw(D_[k,:,:], g_)).to(device)
        R[k,:,:] = Rk
        total_loss = total_loss + Rk[-2,-2]
      ctx.save_for_backward(D, R, gamma)
      return total_loss / batch_size
  
    @staticmethod
    def backward(ctx, grad_output):
      device = grad_output.device
      D, R, gamma = ctx.saved_tensors
      batch_size,N,N = D.shape
      D_ = D.detach().cpu().numpy()
      R_ = R.detach().cpu().numpy()
      g_ = gamma.item()

      E = torch.zeros((batch_size, N ,N)).to(device) 
      for k in range(batch_size):         
        Ek = torch.FloatTensor(compute_softdtw_backward(D_[k,:,:], R_[k,:,:], g_)).to(device)
        E[k:k+1,:,:] = Ek

      return grad_output * E, None


