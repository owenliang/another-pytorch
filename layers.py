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
        onehots=np.eye(probs.shape[-1])[t.data]
        return -(log_probs*onehots).sum()/x.shape[0]

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
    class MLPRegression(Layer):
        def __init__(self):
            super().__init__()
            self.linear1=Linear(1,10)
            self.sigmoid1=Sigmoid()
            self.linear2=Linear(10,1)
        
        def _forward(self,x):   # x: (batch,1), y:(batch,1)
            y=self.linear1(x)
            y=self.sigmoid1(y)
            return self.linear2(y)
    model=MLPRegression()
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

    print('MLP分类测试')
    class MLPClassification(Layer):
        def __init__(self):
            super().__init__()
            self.linear1=Linear(2,10)
            self.sigmoid1=Sigmoid()
            self.linear2=Linear(10,3)
        
        def _forward(self,x): 
            y=self.linear1(x)
            y=self.sigmoid1(y)
            return self.linear2(y)
    
    batch_size=30
    epoch=500

    model=MLPClassification() 
    from optimizers import MomentumSGB 
    optimizer=MomentumSGB(model.params(),lr=0.1)

    from datasets import get_spiral
    import math 
    train_x,train_y=get_spiral(train=True)  
    loss_history=[]
    for i in range(epoch):
        idx=np.random.permutation(len(train_y))
        iters=math.ceil(len(train_y)/batch_size)
        epoch_loss=0
        for j in range(iters):
            batch_idx=idx[batch_size*j:batch_size*(j+1)]
            x=train_x[batch_idx]
            y=train_y[batch_idx]
            output=model(x)
            loss=SoftmaxCrossEntropy1D()(output,y)
            model.zero_grads()
            loss.backward()
            optimizer.step()
            epoch_loss+=float(loss.data)
        loss_history.append(epoch_loss/iters)
        print('epoch:',i,'avg_loss:',loss_history[-1])
    # 绘制Loss曲线
    import matplotlib.pyplot as plt 
    plt.plot(np.arange(epoch),loss_history)
    plt.show()
    # model决策边界图
    x0_min,x0_max=math.floor(train_x[:,0].min()),math.ceil(train_x[:,0].max())
    x1_min,x1_max=math.floor(train_x[:,1].min()),math.ceil(train_x[:,1].max())
    X,Y=np.meshgrid(np.linspace(x0_min,x0_max,1000),np.linspace(x1_min,x1_max,1000))
    points=np.concatenate((X[...,np.newaxis],Y[...,np.newaxis]),axis=2)
    points=points.reshape((1000*1000,2))
    pred_y=model(points)
    pred_y=np.argmax(pred_y.data,axis=-1)
    pred_y=pred_y.reshape((1000,1000,))
    plt.contourf(X,Y,pred_y)    
    # 实际样本分布图
    for c in range(3):  # 3个分类ID的样本分布
        mask=train_y==c
        plt.scatter(train_x[mask,0],train_x[mask,1],) 
    plt.show()