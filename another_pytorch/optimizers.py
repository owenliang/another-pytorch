import numpy as np 
from another_pytorch.torch import get_array_module

class Optimizer:
    def __init__(self,params_iter):
        self.params=list(params_iter)

    def step(self):
        for param in self.params:
            if param.grad is not None:
                self._update(param)

    def _update(self,param):
        raise NotImplementedError()

class SGD(Optimizer):
    def __init__(self,params_iter,lr=0.01):
        super().__init__(params_iter)
        self.lr=lr
    
    def _update(self,param):
        param.data-=self.lr*param.grad.data

# 加速度SGD
class MomentumSGB(Optimizer):
    def __init__(self,params_iter,lr=0.01,momentum=0.9):
        super().__init__(params_iter)
        self.lr=lr
        self.momentum=momentum
        self.param_state={}

    def _update(self,param):
        xp=get_array_module(param.data)
        param_id=id(param)
        if param_id not in self.param_state:
            self.param_state[param_id]=xp.zeros_like(param.data)
        self.param_state[param_id]=self.param_state[param_id]*self.momentum-self.lr*param.grad.data
        param.data+=self.param_state[param_id]

if __name__=='__main__':
    from another_pytorch.layers import *
    class MLP(Layer):
        def __init__(self):
            super().__init__()
            self.linear1=Linear(1,10)
            self.sigmoid1=Sigmoid()
            self.linear2=Linear(10,1)
        
        def _forward(self,x):   # x: (batch,1), y:(batch,1)
            y=self.linear1(x)
            y=self.sigmoid1(y)
            return self.linear2(y)
    model=MLP()
    optimizer=MomentumSGB(model.params(),lr=0.1,momentum=0.9)
    train_x=np.random.randint(1,20,(1000,1))   # x: (1000,1)
    train_y=train_x*5+10
    for iter in range(10000):
        # forward
        pred_y=model(train_x)
        # loss 
        loss=(((pred_y-train_y)/20)**2).sum()/train_x.shape[0]  # mse loss
        # backward
        model.zero_grads()
        loss.backward()
        # optimize
        optimizer.step()
        if iter%500==0:
            print('iter:',iter,'loss:',loss)