# -*- coding: utf-8 -*-
import torch
from torch.autograd.gradcheck import zero_gradients
import numpy as np
from torch.autograd import Variable

def compute_jacobian(inputs, output):
    assert inputs.requires_grad
    num_classes = output.size()[1]
    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()
    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data
    return torch.transpose(jacobian, dim0=0, dim1=1)
def proj(u, v):
    # notice: this algrithm assume denominator isn't zero
    return u * np.dot(v,u) / np.dot(u,u)  

def GS(V):
    V = 1.0 * V     # to float
    U = np.copy(V)
    for i in range(1, V.shape[1]):
        for j in range(i):
            U[:,i] -= proj(U[:,j], V[:,i])
    # normalize column
    den=(U**2).sum(axis=0) **0.5
    E = U/den
    # assert np.allclose(E.T, np.linalg.inv(E))
    return U,den,E

def Comput_Lyapunov(dyn_mdl, init_state, pred_time_steps, init_cov_factor, dt_integration, need_timestep):
    #args : dynamical model, initial state, number of prediction timesteps, init cov factor
    #returns : lyapunov exponents and lyapunov dimension
    #
    tmp = np.reshape(init_state,(1,len(init_state)))
    cov_init = init_cov_factor*np.eye(len(init_state))
    z = np.zeros((pred_time_steps,len(init_state),len(init_state)))
    y = np.zeros((pred_time_steps,len(init_state),len(init_state)))
    l = np.zeros((pred_time_steps,len(init_state)))
    w = np.zeros((pred_time_steps,len(init_state),len(init_state)))
    cov = np.zeros((pred_time_steps,len(init_state),len(init_state)))
    l[0,:] = np.ones((len(init_state)))
    w[0,:,:]=cov_init
    cov[0,:,:]=cov_init
    l_exp  = np.zeros((pred_time_steps,len(init_state)))
    for i in range(1,pred_time_steps):
        tmp = Variable(torch.from_numpy(tmp).float())
        tmp.requires_grad = True
        if need_timestep == True: 
            tmp_out = dyn_mdl(tmp,dt_integration)[0]
        else:
            tmp_out = dyn_mdl(tmp)[0]
        tmp_out = tmp_out.reshape((1,len(init_state)))
        jac = compute_jacobian(tmp, tmp_out)
        z[i,:,:] = np.dot(jac,w[i-1,:,:])[0]
        y[i,:,:],l[i,:],w[i,:,:]=GS(z[i,:,:])
        tmp = np.reshape(tmp_out.data.numpy(),(1,len(init_state)))
        cov[i,:,:] = np.dot(np.dot(jac.data.numpy()[0,:,:],w[i-1,:,:]),jac.data.numpy()[0,:,:].T)
        l_exp[i,:] = np.sum(np.log(l[2:i,:]),axis = 0)/(np.shape(l[2:i,:])[0]-1)/dt_integration
    l_dim = 2+(1./np.abs(l_exp[-1,-1]))*np.sum(l_exp[-1,:2])
    return l_exp, l_dim

def Comput_LyapunovRINN(dyn_mdl, init_state, pred_time_steps, init_cov_factor, dt_integration,order, need_timestep):
    #args : dynamical model, initial state, number of prediction timesteps, init cov factor
    #returns : lyapunov exponents and lyapunov dimension
    #
    tmp = np.reshape(init_state,(1,len(init_state)))
    cov_init = init_cov_factor*np.eye(len(init_state))
    z = np.zeros((pred_time_steps,len(init_state),len(init_state)))
    y = np.zeros((pred_time_steps,len(init_state),len(init_state)))
    l = np.zeros((pred_time_steps,len(init_state)))
    w = np.zeros((pred_time_steps,len(init_state),len(init_state)))
    cov = np.zeros((pred_time_steps,len(init_state),len(init_state)))
    l[0,:] = np.ones((len(init_state)))
    w[0,:,:]=cov_init
    cov[0,:,:]=cov_init
    tmp_out_data = []
    l_exp  = np.zeros((pred_time_steps,len(init_state)))
    for i in range(1,pred_time_steps):
        tmp = Variable(torch.from_numpy(tmp).float())
        tmp.requires_grad = True
        if need_timestep == True: 
            tmp_out = dyn_mdl(tmp,dt_integration, order)[0]
        else:
            tmp_out = dyn_mdl(tmp)[0]
        tmp_out = tmp_out.reshape((1,len(init_state)))
        jac = compute_jacobian(tmp, tmp_out)
        tmp_out_data.append(tmp_out.data.numpy())
        z[i,:,:] = np.dot(jac,w[i-1,:,:])[0]
        y[i,:,:],l[i,:],w[i,:,:]=GS(z[i,:,:])
        tmp = np.reshape(tmp_out.data.numpy(),(1,len(init_state)))
        cov[i,:,:] = np.dot(np.dot(jac.data.numpy()[0,:,:],w[i-1,:,:]),jac.data.numpy()[0,:,:].T)
        l_exp[i,:] = np.sum(np.log(l[2:i,:]),axis = 0)/(np.shape(l[2:i,:])[0]-1)/dt_integration
    l_dim = 2+(1./np.abs(l_exp[-1,-1]))*np.sum(l_exp[-1,:2])
    return l_exp, l_dim, tmp_out_data


def prediction(dyn_mdl, init_state, nb_steps_pred, dt_integration, need_timestep):
    y_pred=np.zeros((nb_steps_pred,len(init_state)))
    tmp = np.reshape(init_state,(1,len(init_state)))
    tmp = Variable(torch.from_numpy(tmp).float())

    for i in range(nb_steps_pred):
        if need_timestep == True: 
            y_pred[i,:] = dyn_mdl(tmp,dt_integration)[0].data.numpy()
        else:
            y_pred[i,:] = dyn_mdl(tmp)[0].data.numpy()
        tmp = Variable(torch.from_numpy(np.reshape(y_pred[i,:] ,(1,len(init_state)))).float())
    return y_pred

def predictionRINN(dyn_mdl, init_state, nb_steps_pred, dt_integration, order, need_timestep):
    y_pred=np.zeros((nb_steps_pred,len(init_state)))
    tmp = np.reshape(init_state,(1,len(init_state)))
    tmp = Variable(torch.from_numpy(tmp).float())

    for i in range(nb_steps_pred):
        if need_timestep == True: 
            y_pred[i,:] = dyn_mdl(tmp, dt_integration, order)[0].data.numpy()
        else:
            y_pred[i,:] = dyn_mdl(tmp)[0].data.numpy()
        tmp = Variable(torch.from_numpy(np.reshape(y_pred[i,:] ,(1,len(init_state)))).float())
    return y_pred
    
def RMSE(a,b):
    """ Compute the Root Mean Square Error between 2 n-dimensional vectors. """
 
    return np.sqrt(np.mean((a-b)**2))

def normalise(M):
    """ Normalize the entries of a multidimensional array sum to 1. """

    c = np.sum(M);
    # Set any zeros to one before dividing
    d = c + 1*(c==0);
    M = M/d;
    return M;

def mk_stochastic(T):
    """ Ensure the matrix is stochastic, i.e., the sum over the last dimension is 1. """

    if len(T.shape) == 1:
        T = normalise(T);
    else:
        n = len(T.shape);
        # Copy the normaliser plane for each i.
        normaliser = np.sum(T,n-1);
        normaliser = np.dstack([normaliser]*T.shape[n-1])[0];
        # Set zeros to 1 before dividing
        # This is valid since normaliser(i) = 0 iff T(i) = 0

        normaliser = normaliser + 1*(normaliser==0);
        T = T/normaliser.astype(float);
    return T;

def sample_discrete(prob, r, c):
    """ Sampling from a non-uniform distribution. """

    # this speedup is due to Peter Acklam
    cumprob = np.cumsum(prob);
    n = len(cumprob);
    R = np.random.rand(r,c);
    M = np.zeros([r,c]);
    for i in range(0,n-1):
        M = M+1*(R>cumprob[i]);    
    return int(M)

def resampleMultinomial(w):
    """ Multinomial resampler. """

    M = np.max(w.shape);
    Q = np.cumsum(w,0);
    Q[M-1] = 1; # Just in case...
    i = 0;
    indx = [];
    while (i<=(M-1)):
        sampl = np.random.rand(1,1);
        j = 0;
        while (Q[j]<sampl):
            j = j+1;
        indx.append(j);
        i = i+1
    return indx

def inv_using_SVD(Mat, eigvalMax):
    """ SVD decomposition of Matrix. """
    
    U,S,V = np.linalg.svd(Mat, full_matrices=True);
    eigval = np.cumsum(S)/np.sum(S);
    # search the optimal number of eigen values
    i_cut_tmp = np.where(eigval>=eigvalMax)[0];
    S = np.diag(S);
    V = V.T;
    i_cut = np.min(i_cut_tmp)+1;
    U_1 = U[0:i_cut,0:i_cut];
    U_2 = U[0:i_cut,i_cut:];
    U_3 = U[i_cut:,0:i_cut];
    U_4 = U[i_cut:,i_cut:];
    S_1 = S[0:i_cut,0:i_cut];
    S_2 = S[0:i_cut,i_cut:];
    S_3 = S[i_cut:,0:i_cut];
    S_4 = S[i_cut:,i_cut:];
    V_1 = V[0:i_cut,0:i_cut];
    V_2 = V[0:i_cut,i_cut:];
    V_3 = V[i_cut:,0:i_cut];
    V_4 = V[i_cut:,i_cut:];
    tmp1 = np.dot(np.dot(V_1,np.linalg.inv(S_1)),U_1.T);
    tmp2 = np.dot(np.dot(V_1,np.linalg.inv(S_1)),U_3.T);
    tmp3 = np.dot(np.dot(V_3,np.linalg.inv(S_1)),U_1.T);
    tmp4 = np.dot(np.dot(V_3,np.linalg.inv(S_1)),U_3.T);
    inv_Mat = np.concatenate((np.concatenate((tmp1,tmp2),axis=1),np.concatenate((tmp3,tmp4),axis=1)),axis=0);
    tmp1 = np.dot(np.dot(U_1,S_1),V_1.T);
    tmp2 = np.dot(np.dot(U_1,S_1),V_3.T);
    tmp3 = np.dot(np.dot(U_3,S_1),V_1.T);
    tmp4 = np.dot(np.dot(U_3,S_1),V_3.T);
    hat_Mat = np.concatenate((np.concatenate((tmp1,tmp2),axis=1),np.concatenate((tmp3,tmp4),axis=1)),axis=0);
    det_inv_Mat = np.prod(np.diag(S[0:i_cut,0:i_cut]));   
    return inv_Mat;