from torch import *
import numpy as np 

class Layer:
    def __init__(self):
        self.param_names=set()
        self.cuda=False
        self.eval_mode=False # train mode by default

    def __call__(self,*inputs):
        inputs=[to_variable(var_or_data,self.cuda) for var_or_data in inputs]
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

    def named_params(self,prefix=''):
        for name in self.param_names:
            value=getattr(self,name)
            fullname=prefix+'.'+name if prefix else name
            if isinstance(value,Parameter):
                yield fullname,value
            else:
                yield from value.named_params(fullname)

    def sublayers(self):
        for name in self.param_names:
            value=getattr(self,name)
            if isinstance(value,Layer):
                yield value
                yield from value.sublayers()

    def zero_grads(self):
        for param in self.params():
            param.zero_grad()

    def to_cuda(self):
        self.cuda=True
        for p in self.params():
            p.to_cuda()
        for l in self.sublayers():
            l.cuda=True
        return self
    
    def to_cpu(self):
        self.cuda=False
        for p in self.params():
            p.to_cpu()
        for l in self.sublayers():
            l.cuda=False
        return self 

    def save_weights(self,path):
        param_dict={name:to_numpy(param.data) for name,param in self.named_params()}
        np.savez_compressed(path,**param_dict)
    
    def load_weights(self,path):
        param_dict=np.load(path+'.npz')
        for name,param in self.named_params():
            param.data=param_dict[name]
            if self.cuda:
                param.to_cuda()

    def train(self):
        self.eval_mode=False
        for l in self.sublayers():
            l.eval_mode=False

    def eval(self):
        self.eval_mode=True
        for l in self.sublayers():
            l.eval_mode=True

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

class Dropout(Layer):
    def __init__(self,p=0.5):
        super().__init__()
        self.p=p 
    
    def _forward(self,x):
        if self.eval_mode:
            return x
        else:
            xp=get_array_module(x.data)
            mask=xp.random.rand(*x.shape)>self.p
            return x*mask/(1-self.p)

class SoftmaxCrossEntropy1D(Layer):    
    def _forward(self,x,t):
        probs=Softmax1D()(x)
        log_probs=Log()(Clip(1e-15,1.0)(probs))  # for ln(x), x can not be 0
        onehots=np.eye(probs.shape[-1])[to_numpy(t.data)]
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

    print('Layer sublayers()测试')
    for layer in parent_linear.sublayers():
        print('sublayer:',layer)

    print('Layer named_params()测试')
    model=ParentLayer()
    for name,param in model.named_params():
        print('model:',name,param)

    print('Layer save_weights&load_weights测试')
    model.save_weights('model.pt')
    loaded_model=ParentLayer()
    loaded_model.load_weights('model.pt')
    for name,param in loaded_model.named_params():
        print('loaded_model:',name,param)

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

    print('Dropout()测试')
    x=np.ones((3,3))
    dropout=Dropout(p=0.70)
    y=dropout(x)
    print('dropout:',y)
    dropout.eval()
    y=dropout(x)
    print('dropout in eval:',y)

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

    from dataset import get_spiral,SpiralDataset
    from dataloader import DataLoader
    import math 
    train_x,train_y=get_spiral(train=True)  
    dataset=SpiralDataset()
    dataloader=DataLoader(dataset,batch_size=30)
    loss_history=[]
    for i in range(epoch):
        iters=math.ceil(len(train_y)/batch_size)
        epoch_loss=0
        for x,y in dataloader:
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

    print('MNIST分类测试')
    class MNIST_MLP(Layer):
        def __init__(self):
            super().__init__()
            self.linear1=Linear(28*28,1000)
            self.relu1=Relu()
            self.linear2=Linear(1000,10)
        
        def _forward(self,x): 
            y=self.linear1(x)
            y=self.relu1(y)
            return self.linear2(y)
        
    def img_transformer(x):
        x=x.flatten()
        x=x/256.0
        return x 
    
    # evaluation
    def accuracy(output,t):
        pred_t=output.data.argmax(axis=-1)
        acc=(pred_t==t.data).sum()/t.shape[0]
        return Variable(acc)

    from dataset import MNISTDataset
    train_dataset=MNISTDataset(train=True,transformer=img_transformer)
    test_dataset=MNISTDataset(train=False,transformer=img_transformer)
    
    epoch=5
    batch_size=100
    model=MNIST_MLP()
    optimizer=MomentumSGB(model.params(),lr=0.1)
    loss_fn=SoftmaxCrossEntropy1D()
    
    train_dataloader=DataLoader(train_dataset,batch_size)
    test_dataloader=DataLoader(test_dataset,batch_size)
    for e in range(epoch):
        for x,t in train_dataloader:
            output=model(x)
            loss=loss_fn(output,t)
            model.zero_grads()
            loss.backward()
            optimizer.step()
            print('loss:',loss,'acc:',accuracy(output,t))

    print('[CUDA]MNIST分类测试')
    try:
        from dataset import MNISTDataset
        train_dataset=MNISTDataset(train=True,transformer=img_transformer)
        test_dataset=MNISTDataset(train=False,transformer=img_transformer)
        
        epoch=5
        batch_size=100
        model=MNIST_MLP().to_cuda()
        optimizer=MomentumSGB(model.params(),lr=0.1)
        loss_fn=SoftmaxCrossEntropy1D().to_cuda()
        
        train_dataloader=DataLoader(train_dataset,batch_size)
        test_dataloader=DataLoader(test_dataset,batch_size)
        for e in range(epoch):
            for x,t in train_dataloader:
                x=x.to_cuda()
                t=t.to_cuda() 
                output=model(x)

                loss=loss_fn(output,t)
                model.zero_grads()
                loss.backward()
                optimizer.step()
                print('loss:',loss,'acc:',accuracy(output,t))
    except Exception as e:
        print('没有NVIDIA显卡,',e)