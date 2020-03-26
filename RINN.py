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
    torch.manual_seed(1234)
    np.random.seed(1234)    
    class FC_net(torch.nn.Module):
        def __init__(self, params):
            super(FC_net, self).__init__()
            self.linearCell   = torch.nn.Linear(params['dim_input'], params['dim_hidden_linear']) 
            self.BlinearCell1 = torch.nn.ModuleList([torch.nn.Linear(params['dim_input'], 1) for i in range(params['bi_linear_layers'])])
            self.BlinearCell2 = torch.nn.ModuleList([torch.nn.Linear(params['dim_input'], 1) for i in range(params['bi_linear_layers'])])
            augmented_size    = params['bi_linear_layers'] + params['dim_hidden_linear']
            self.transLayers = torch.nn.ModuleList([torch.nn.Linear(augmented_size, params['dim_output'])])
            self.transLayers.extend([torch.nn.Linear(params['dim_output'], params['dim_output']) for i in range(1, params['transition_layers'])])
            self.outputLayer  = torch.nn.Linear(params['dim_output'], params['dim_output']) 
        def forward(self, inp):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            BP_outp = Variable(torch.zeros((inp.size()[0],params['bi_linear_layers'])))
            L_outp   = self.linearCell(inp)
            for i in range((params['bi_linear_layers'])):
                BP_outp[:,i]=self.BlinearCell1[i](inp)[:,0]*self.BlinearCell2[i](inp)[:,0]
            aug_vect = torch.cat((L_outp, BP_outp), dim=1)
            for i in range((params['transition_layers'])):
                aug_vect = (self.transLayers[i](aug_vect))
            grad = self.outputLayer(aug_vect)
            return grad
    model  = FC_net(params)
    class INT_net(torch.nn.Module):
        def __init__(self, params,order):
            super(INT_net, self).__init__()
            self.Dyn_net = model
            a = np.tril(np.random.uniform(size=(params['dim_observations'],order,order)),k=-1)
            b = np.random.uniform(size=(params['dim_observations'],order))
            c = np.random.uniform(size=(params['dim_observations'],order))
            self.a = torch.nn.Parameter(torch.from_numpy(a[:,:,:]).float())
            self.b = torch.nn.Parameter(torch.from_numpy(b).float())
            self.c = torch.nn.Parameter(torch.from_numpy(c).float())

        def forward(self, inp, dt, order):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            k = [(self.Dyn_net(inp))]
            for i in range(1,order):
                coef_sum = torch.autograd.Variable(torch.zeros(k[0].size()))
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
            pred_sum = torch.autograd.Variable(torch.zeros(k[0].size()))
            for i in range(0,order): 
                pred_sum = pred_sum+k[i]*self.b[:,i].expand_as(k[i])
            pred = inp +dt*pred_sum
            return pred ,k[0], inp

    x = Variable(torch.from_numpy(X_train).float())
    y = Variable(torch.from_numpy(Y_train).float())
    z = Variable(torch.from_numpy(Grad_t).float())
    # Construct our model by instantiating the class defined above
    
    modelRINN = INT_net(params,order)
    # Construct our loss function and an Optimizer. The call to model.parameters()

    if pretrained :
        modelRINN.load_state_dict(torch.load(path))
    criterion = torch.nn.MSELoss(reduction = 'elementwise_mean')
    optimizer = torch.optim.Adam(modelRINN.parameters(), lr = params['lr'],weight_decay=0.1)
    optimizer.param_groups[0]['params'].append(modelRINN.a)
    optimizer.param_groups[0]['params'].append(modelRINN.b)
    optimizer.param_groups[0]['params'].append(modelRINN.c)    

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
        print(t,loss)

    print ('Learning prediction model')   
    modelRINN.apply(clipper)     
    for t in range(params['ntrain'][1]):
        for b in range(x.shape[0]):
        # Forward pass: Compute predicted states by passing x to the model
            pred ,grad , inp = modelRINN(x[b,:,:],params['dt_integration'],order)
            # Compute and print loss
            loss1 = criterion(pred, y[b,:,:])
            #loss2 = criterion(grad, z[b,:,:])
            loss = 1.0*loss1
            if loss.data.numpy()<1000 and not (np.isnan(loss.data.numpy())):
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(modelRINN.parameters(),5)
                #modelRINN.apply(clipper)
                if t % clipper.frequency == 0:
                    modelRINN.apply(clipper)     
        print(t,loss)        
    return modelRINN
