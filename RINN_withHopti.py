#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 21:02:22 2018

@author: saidouala
"""

import numpy as np 
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def RINN_model(X_train, Y_train, Grad_t, params, order, pretrained, path):
    
    class applyRK_Constraints(object):
        def __init__(self, frequency=1):
            self.frequency = frequency
    
        def __call__(self, module):
            if hasattr(module, 'b'):
                module.b.data = (torch.abs(module.b.data))
                module.b.data  =  ((module.b.data) / (module.b.data).sum(1,keepdim = True).expand_as(module.b.data))
            if hasattr(module, 'c'):
                module.c.data = module.c.data
                module.c.data[:,0] = 0
                module.c.data = module.c.data.sub_(torch.min(module.c.data)).div_(torch.max(module.c.data) - torch.min(module.c.data)).sort()[0]
                 
    class FC_net(torch.nn.Module):
        def __init__(self, params):
            super(FC_net, self).__init__()
            self.coef_mdl =(params['lin_coef'])
        def forward(self, inp):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            grad = self.coef_mdl*inp
            return grad
    model  = FC_net(params)
    torch.manual_seed(55)
    np.random.seed(55)
    class INT_net(torch.nn.Module):
        def __init__(self, params,order):
            super(INT_net, self).__init__()
            self.Dyn_net = model
            a = np.tril(np.random.uniform(size=(params['dim_observations'],order,order)),k=-1)
            b = np.random.uniform(size=(params['dim_observations'],order))
            c = np.random.uniform(size=(params['dim_observations'],order))
            self.a = torch.nn.Parameter(torch.from_numpy(a[:,:,:]).double())
            self.b = torch.nn.Parameter(torch.from_numpy(b).double())
            self.c = torch.nn.Parameter(torch.from_numpy(c).double())

        def forward(self, inp, dt, order):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            k = [(self.Dyn_net(inp))]
            for i in range(1,order):
#                coef_sum = torch.autograd.Variable(torch.zeros(k[0].size()).double())
                coef_sum = 0
                for j in range(0,i):
                    if j ==0:
                        if i == 1:
                            coef_sum = coef_sum + k[j]*(self.c[:,i]).expand_as(k[j])
                        else:
                            coef_sum = coef_sum + k[j]*(self.c[:,i]-self.a[:,i,1:i].sum(1)).expand_as(k[j])
                    else :
                        coef_sum = coef_sum + k[j]*self.a[:,i,j].expand_as(k[j])
                rk_inp = inp+dt*coef_sum        
                k.append(self.Dyn_net(rk_inp))
#            pred_sum = torch.autograd.Variable(torch.zeros(k[0].size()).double())
            pred_sum = 0   
            for i in range(0,order): 
                pred_sum = pred_sum+k[i]*self.b[:,i].expand_as(k[i])
            pred = inp +dt*pred_sum
            gain = pred/inp
            return pred ,k[0], gain
    
    model2 = INT_net(params,order)










    x = Variable(torch.from_numpy(X_train).double())
    y = Variable(torch.from_numpy(Y_train).double())
    z = Variable(torch.from_numpy(Grad_t).double())
    
    hi = np.arange(0,6+0.000001,0.01)
    h_int = Variable(torch.from_numpy(hi).double())
    gain_true = Variable(torch.from_numpy(np.ones_like(h_int)).double())
    
    # Construct our model by instantiating the class defined above

    modelRINN = INT_net(params,order)
    # Construct our loss function and an Optimizer. The call to model.parameters()

    if pretrained :
        modelRINN.load_state_dict(torch.load(path))
    criterion = torch.nn.MSELoss(reduction = 'sum')
    optimizer = torch.optim.Adam(modelRINN.parameters(), lr = 0.001)
    optimizer.param_groups[0]['params'].append(modelRINN.a)
    optimizer.param_groups[0]['params'].append(modelRINN.b)
    optimizer.param_groups[0]['params'].append(modelRINN.c)    
    loss_hist = []
    clipper = applyRK_Constraints()    
    print ('Learning dynamical model')
    for t in range(params['ntrain'][0]):
        for b in range(x.shape[0]):
            # Forward pass: Compute predicted gradients by passing x to the model
            pred ,grad , inp = modelRINN(x[b,:,:],params['dt_integration'],order)
            # Compute and print loss
            loss = criterion(grad, z[b,:,:])
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print(t,loss.data[0])    

    print ('Learning prediction model')   
    #modelRINN.apply(clipper)     
    for t in range(params['ntrain'][1]):
        for b in range(x.shape[0]):
        # Forward pass: Compute predicted states by passing x to the model
            pred ,grad , inp = modelRINN(x[b,:,:],params['dt_integration'],order)
            # Compute and print loss
            loss = criterion(pred, y[b,:,:])

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad() 
            modelRINN.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(modelRINN.parameters(),5)
            optimizer.step()
            modelRINN.apply(clipper)
#            if t % clipper.frequency == 0:
#                modelRINN.apply(clipper)    
        loss_hist.append(loss.data.numpy())
        print(t,loss)


    class Gain(torch.nn.Module):
        def __init__(self):
            super(Gain, self).__init__()
            self.Pred_net_RINN = modelRINN

        def forward(self, h_int, inp, order):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            gain = Variable(torch.zeros(h_int.size()).double())
            for i in range(h_int.shape[0]):
                pred ,tmp, tmpp =self.Pred_net_RINN(inp, h_int[i], order)
                gain[i] = tmpp[-1,0]
            return pred ,gain

    model_int = Gain()
#    model_init = Gain()

    # Construct our loss function and an Optimizer. The call to model.parameters()

    def customized_loss(x, y):
        gain_to_zero = F.relu((x-y))**2
        #for i in range(len(gain_to_zero)):
        #    if gain_to_zero[i]>0:
          #      gain_to_zero[i] = F.relu(gain_to_zero[i].clone())
#        loss = torch.max(gain_to_zero,comp)
        
        return torch.mean(gain_to_zero)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model_int.parameters())
    optimizer.param_groups[0]['params'].append(model_int.Pred_net_RINN.a)
    optimizer.param_groups[0]['params'].append(model_int.Pred_net_RINN.b)
    optimizer.param_groups[0]['params'].append(model_int.Pred_net_RINN.c)    
    loss_hist = []        
    print ('optim integration model')
    for t in range(params['ntrain'][2]):
        for b in range(x.shape[0]):
            # Forward pass: Compute predicted gradients by passing x to the model
            pred ,gain_op = model_int(h_int, x[b,:,:],order)
            # Compute and print loss
            loss = customized_loss((gain_op), gain_true)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print(t,loss.data[0])
#    plt.plot((torch.abs(gain_op)).data.numpy())
#    plt.plot((torch.abs(gain_op2)).data.numpy())    
    return model, modelRINN, np.array(loss_hist), model_int
