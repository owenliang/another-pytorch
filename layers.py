from torch import *
import numpy as np 

class Layer:
    def __init__(self):
        self.param_names=set()

    def __call__(self,*inputs):
        inputs=[to_variable(var_or_data) for var_or_data in inputs]
        return self._forward(*inputs)
    
    def __setattr__(self,name,value):
        if isinstance(value,Parameter) or isinstance(value,Layer):
            self.param_names.add(name)
        super().__setattr__(name,value)

    def params(self):
        for name in self.param_names:
            value=getattr(self,name)
            if isinstance(value,Parameter):
                yield value 
            elif isinstance(value,Layer):
                yield from value.params()

    def zero_grads(self):
        for param in self.params():
            param.zero_grad()

    def _forward(self,*inputs):
        raise NotImplementedError()

class Linear(Layer):
    def __init__(self,in_size,out_size):
        super().__init__()
        self.w=Parameter(np.random.randn(in_size,out_size)/np.sqrt(in_size),name='W')
        self.b=Parameter(np.zeros(out_size),name='b')

    def _forward(self,x):
        return x@self.w+self.b

class Sigmoid(Layer):
    def _forward(self,x):
        return 1/(1+Exp()(-x))

class Softmax1D(Layer):
    def _forward(self,x):
        x_exp=Exp()(x)
        return x_exp/x_exp.sum(axes=-1,keepdims=True)

class SoftmaxCrossEntropy1D(Layer):    
    def _forward(self,x,t):
        probs=Softmax1D()(x)
        log_probs=Log()(Clip(1e-15,1.0)(probs))  # for ln(x), x can not be 0
        return -log_probs[np.arange(0,x.shape[0]),t.data].sum()/x.shape[0]  # TODO: log_probs[...,t.data]结果就不对,必须arange指定行号,原因需要再看下

if __name__=='__main__':
    print('Linear测试')
    linear=Linear(5,3)
    x=np.random.rand(100,5)
    y=linear(x)
    y.backward()
    print('y:',y.shape)
    print('linear w grad:',linear.w.grad.shape)
    print('linear b grad:',linear.b.grad.shape)

    print('Layer params()测试')
    class ChildLayer(Layer):
        def __init__(self):
            super().__init__()
            self.linear=Linear(3,5)
    class ParentLayer(Layer):
        def __init__(self):
            super().__init__()
            self.child=ChildLayer()
            self.linear=Linear(5,3)
    parent_linear=ParentLayer()
    for param in parent_linear.params():
        print('param:',param.name,param.data.shape)

    print('Sigmoid()测试')
    sigmoid=Sigmoid()
    x=Variable(np.random.randint(5,10,(2,5)))
    print('x:',x)
    y=sigmoid(x)
    y.backward()
    print('y:',y)
    print('x grad:',x.grad.shape)

    print('Softmax1D()测试')
    softmax=Softmax1D()
    x=Variable([[1,1,2],[4,1,5]])
    probs=softmax(x)
    print('softmax:',probs)

    print('SoftmaxCrossEntropy1D()测试')
    crossentropy_loss=SoftmaxCrossEntropy1D()
    x=Variable([[0,0,1],[1,0,0]])
    t=Variable([2,0])
    loss=crossentropy_loss(x,t)
    print('loss:',loss)

    print('MLP回归测试')
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
    from optimizers import SGD 
    optimizer=SGD(model.params(),lr=0.1)
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