import numpy as np 
from graphviz import Digraph
from contextlib import contextmanager
try:
    import cupy as cp
except:
    HAS_CUDA=False
else:
    HAS_CUDA=True

class Variable:
    def __init__(self,data,name=None):
        if get_array_module(data)!=np:
            self.cuda=True
            self.data=cp.asarray(data)  # asarray won't copy ndarray if data is already ndarray
        else:
            self.cuda=False 
            self.data=np.asarray(data)  # asarray won't copy ndarray if data is already ndarray
        self.name=name  # str
        self.grad=None  # Variable
        self.func=None  # Function
        self.gen=0      # Generation

    def backward(self):
        xp=get_array_module(self.data)
        self.grad=Variable(xp.ones_like(self.data)) # if data is in cuda, grad should be in cuda too
                
        func_q=[self.func]
        func_set=set(func_q)
        while len(func_q)!=0:
            f=func_q.pop()
            f.backward()
            for var in f.inputs:
                if var.func is None or var.func in func_set:
                    continue 
                func_set.add(var.func)
                func_q.append(var.func)
            func_q=sorted(func_q,key=lambda f:f.gen)

    def zero_grad(self):
        self.grad=None

    def __repr__(self):
        return str(self.data)
    
    def __add__(self,other):
        return Add()(self,other)

    def __radd__(self,other):
        return Add()(other,self)
    
    def __sub__(self,other):
        return Sub()(self,other)
    
    def __rsub__(self,other):
        return Sub()(other,self)

    def __mul__(self,other):
        return Mul()(self,other)

    def __rmul__(self,other):
        return Mul()(other,self)
    
    def __truediv__(self,other):
        return Div()(self,other)

    def __rtruediv__(self,other):
        return Div()(other,self)
    
    def __pow__(self,other):
        return Pow(other)(self)
    
    def __neg__(self):
        return Neg()(self)
    
    def __matmul__(self,other): # @
        return MatMul()(self,other)
    
    def __rmatmul__(self,other): # @
        return MatMul()(other,self)
    
    def __getitem__(self,slice):
        return Slice(slice)(self)

    def reshape(self,shape):
        return Reshape(shape)(self)
    
    def transpose(self,axes=None):
        return Transpose(axes)(self)

    def sum(self,axes=None,keepdims=False):
        return Sum(axes,keepdims)(self)

    def broadcast(self,shape):
        return Broadcast(shape)(self)

    @property
    def T(self):
        return Transpose(None)(self)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def to_cuda(self):
        self.cuda=True 
        self.data=to_cupy(self.data)
        return self

    def to_cpu(self):
        self.cuda=False 
        self.data=to_numpy(self.data)
        return self

# for Layer trainable variable
class Parameter(Variable):
    pass 

def to_variable(data,to_cuda):
    if isinstance(data,Variable):
        return data 
    var=Variable(data)
    if to_cuda:
        var.to_cuda()
    return var

def to_numpy(data):
    if not HAS_CUDA:
        return np.asarray(data)
    return cp.asnumpy(data)

def to_cupy(data):
    if not HAS_CUDA:
        raise Exception('HAS_CUDA is False')
    return cp.asarray(data)

def get_array_module(arr):
    if not HAS_CUDA:
        return np
    return cp.get_array_module(arr)

class Function:
    def __init__(self):
        self.gen=None  # max(inputs' generation)
        self.inputs=None
        self.outputs=None

    def _check_inputs(self,inputs):
        has_np=False
        has_cuda=False
        for var_or_data in inputs:
            if isinstance(var_or_data,Variable):
                if var_or_data.cuda:
                    has_cuda=True
                else:
                    has_np=True
        if has_np and has_cuda:
            raise Exception('Function inputs have both numpy and cupy Variable, please check!')
        return has_cuda
    
    def forward(self,*inputs): # Variable
        to_cuda=self._check_inputs(inputs)
        inputs=[to_variable(var_or_data,to_cuda) for var_or_data in inputs] # Variable
        outputs=self._forward(*[var.data for var in inputs])       # ndarray
        if not isinstance(outputs,tuple):
            outputs=outputs,
        
        outputs=[Variable(data) for data in outputs]
        self.gen=max([var.gen for var in inputs])
        for var in outputs:
            var.func=self
            var.gen=self.gen+1

        # store inputs&outputs for backward
        if not NO_GRAD:
            self.inputs=inputs
            self.outputs=outputs
        return outputs[0] if len(outputs)==1 else outputs
    
    __call__=forward

    def backward(self):   
        output_grads=[var.grad for var in self.outputs]    # Variable
        input_grads=self._backward(*output_grads)
        if not isinstance(input_grads,tuple):
            input_grads=input_grads,
        for i in range(len(self.inputs)):
            if self.inputs[i].grad is None:
                self.inputs[i].grad=input_grads[i]
            else:
                self.inputs[i].grad=self.inputs[i].grad+input_grads[i]  # Variable+Variable
        for var in self.outputs:        # clear memory & prepare for double backward
           var.grad=None

    # Overwrite
    def _forward(self,*inputs):    # ndarray
        raise NotImplementedError()
    
    # Overwrite
    def _backward(self,*grad):   # Variable
        raise NotImplementedError()
    
# +
class Add(Function):
    def _forward(self,a,b):
        return a+b

    def _backward(self,grad):
        a_grad,b_grad=grad*1,grad*1
        if self.inputs[0].shape!=grad.shape:
            a_grad=DeBroadcast(self.inputs[0].shape)(grad)
        if self.inputs[1].shape!=grad.shape:
            b_grad=DeBroadcast(self.inputs[1].shape)(grad)
        return a_grad,b_grad

# -
class Sub(Function):
    def _forward(self,a,b):
        return a-b
    
    def _backward(self,grad):
        a_grad,b_grad=grad*1,grad*-1
        if self.inputs[0].shape!=grad.shape:
            a_grad=DeBroadcast(self.inputs[0].shape)(grad)
        if self.inputs[1].shape!=grad.shape:
            b_grad=DeBroadcast(self.inputs[1].shape)(grad)
        return a_grad,b_grad

# *
class Mul(Function):
    def _forward(self,a,b):
        return a*b

    def _backward(self,grad):
        a_grad=self.inputs[1]*grad 
        b_grad=self.inputs[0]*grad
        if self.inputs[0].shape!=grad.shape:
            a_grad=DeBroadcast(self.inputs[0].shape)(a_grad)
        if self.inputs[1].shape!=grad.shape:
            b_grad=DeBroadcast(self.inputs[1].shape)(b_grad)
        return a_grad,b_grad

# /   
class Div(Function):
    def _forward(self,a,b):
        return a/b
    
    def _backward(self,grad):
        a_grad=grad*1/self.inputs[1]
        b_grad=-1*grad*self.inputs[0]/(self.inputs[1]**2)
        if self.inputs[0].shape!=grad.shape:
            a_grad=DeBroadcast(self.inputs[0].shape)(a_grad)
        if self.inputs[1].shape!=grad.shape:
            b_grad=DeBroadcast(self.inputs[1].shape)(b_grad)
        return a_grad,b_grad

# **
class Pow(Function):
    def __init__(self,b):
        self.b=b    # int

    def _forward(self,a):
        xp=get_array_module(a)   # CUDA compatibility
        return xp.power(a,self.b)

    def _backward(self,grad):
        return grad*self.b*self.outputs[0]/self.inputs[0]

# -
class Neg(Function):
    def _forward(self,x):
        return -x 

    def _backward(self,grad):
        return grad*-1

# Matrix Reshape
class Reshape(Function):
    def __init__(self,output_shape):
        self.output_shape=output_shape

    def _forward(self,x):
        xp=get_array_module(x)   # CUDA compatibility
        self.x_shape=x.shape
        return xp.reshape(x,self.output_shape)

    def _backward(self,grad):
        return Reshape(self.x_shape)(grad)

# Matrix Transpose
class Transpose(Function):
    def __init__(self,axes):
        self.axes=axes

    def _forward(self,x):
        xp=get_array_module(x)   # CUDA compatibility
        return xp.transpose(x,self.axes)

    def _backward(self,grad):
        return Transpose(self.axes)(grad)

# Matrix Sum
class Sum(Function):
    def __init__(self,axes,keepdims):
        self.axes=axes
        self.keepdims=keepdims
    
    def _forward(self,x):
        xp=get_array_module(x)   # CUDA compatibility
        self.x_shape=x.shape
        return xp.sum(x,axis=self.axes,keepdims=self.keepdims)

    def _backward(self,grad):   
        # Case1: (3,4,2) -> sum(axes:(0,2),keepdims=True) -> (1,4,1)
        # Case2: (3,4,2) -> sum(axes:(0,2),keepdims=False)-> (4,)
        grad_shape=list(grad.data.shape)
        if len(grad_shape)!=len(self.x_shape):  # keepdims=False
            axes=list(range(len(self.x_shape))) if self.axes is None else self.axes
            for idim in axes:  # (4,) -> (1,4,) -> (1,4,1)
                grad_shape.insert(idim,1)
        
        grad=grad.reshape(grad_shape)  # Reshape function  ,    (1,4,1)
        grad=grad.broadcast(self.x_shape)   # Broadcast function    (1,4,1)->(3,4,2)
        return grad

# Matrix DeBroadcast
class DeBroadcast(Function):
    def __init__(self,output_shape):
        self.output_shape=output_shape

    def _forward(self,x):   # x:(3,4,2) , output_shape: (4,1)
        xp=get_array_module(x)   # CUDA compatibility

        self.x_shape=x.shape
        prefix_ndim=len(x.shape)-len(self.output_shape)  # len((3,4,2))-len((4,1))
        dims=[]
        for idim in range(len(self.output_shape)):
            if self.output_shape[idim]!=x.shape[prefix_ndim+idim]:
                dims.append(prefix_ndim+idim)
        prefix_dims=list(range(prefix_ndim))
        output=xp.sum(x,axis=tuple(prefix_dims+dims),keepdims=True)
        return xp.squeeze(output,axis=tuple(prefix_dims))

    def _backward(self,grad):   # grad:(4,1), return: (3,4,2)
        return Broadcast(self.x_shape)(grad)

# Matrix Broadcast
class Broadcast(Function):
    def __init__(self,output_shape):
        self.output_shape=output_shape 
    
    def _forward(self,x):   
        xp=get_array_module(x)   # CUDA compatibility
        self.x_shape=x.shape
        # Case1: Simple version,  (1,4,1)   -> (3,4,2)
        # Case2: Hard version, (4,1)  ->    (3,4,2)
        return xp.broadcast_to(x,self.output_shape)

    def _backward(self,grad):    # Case2: Hard version,   grad: (3,4,2)  -> (4,1)
        return DeBroadcast(self.x_shape)(grad)

# Matrix Multiply
class MatMul(Function):
    def _forward(self,a,b): # (A,B)X(B,C)->(A,C)
        xp=get_array_module(a)   # CUDA compatibility
        return xp.dot(a,b)

    def _backward(self,grad):
        grad_a=MatMul()(grad,self.inputs[1].transpose())    # (A,C)X(B,C).T
        grad_b=MatMul()(self.inputs[0].transpose(),grad)   # (A,B).TX(A,C)->(B,C)
        return grad_a,grad_b

# e^x
class Exp(Function):
    def _forward(self,x):
        xp=get_array_module(x)   # CUDA compatibility
        return xp.exp(x)
    
    def _backward(self,grad):
        return self.outputs[0]*grad

# Slice like a[2:],a[1],a[3:,1:3],
class Slice(Function):
    def __init__(self,slice):
        self.slice=slice

    def _forward(self,x):
        self.x_shape=x.shape
        return x[self.slice]
    
    def _backward(self,grad):
        return SliceGrad(self.x_shape,self.slice)(grad)
    
class SliceGrad(Function):
    def __init__(self,x_shape,slice):
        self.x_shape=x_shape
        self.slice=slice
    
    def _forward(self,grad):
        xp=get_array_module(grad)   # CUDA compatibility
        grad_x=xp.zeros(self.x_shape)
        xp.add.at(grad_x,self.slice,grad)
        return grad_x

    def _backward(self,grad):
        return Slice(self.slice)(grad)

# Log
class Log(Function):
    def _forward(self,x):
        xp=get_array_module(x)   # CUDA compatibility
        return xp.log(x)
    
    def _backward(self,grad):
        return grad/self.inputs[0]

# Clip(data range limit)
class Clip(Function):
    def __init__(self,x_min,x_max):
        self.x_min=x_min
        self.x_max=x_max 

    def _forward(self,x):
        xp=get_array_module(x)   # CUDA compatibility
        return xp.clip(x,self.x_min,self.x_max)
    
    def _backward(self,grad):
        xp=get_array_module(grad)   # CUDA compatibility
        mask=(self.inputs[0].data>=self.x_min)*(self.inputs[0].data<=self.x_max)
        return grad*mask.astype(xp.uint8)

# Relu activation
class Relu(Function):
    def _forward(self,x):
        xp=get_array_module(x)   # CUDA compatibility
        return xp.maximum(x,xp.zeros(x.shape))
    
    def _backward(self,grad):
        x_grad=grad*(self.inputs[0].data>0)
        return x_grad

# Max
class Max(Function):
    def __init__(self,axis=None,keepdims=False):
        self.axis=axis
        self.keepdims=keepdims

    def _forward(self,x):
        xp=get_array_module(x)
        return xp.max(x,axis=self.axis,keepdims=self.keepdims)
    
    def _backward(self,grad):
        x=self.inputs[0]
        xp=get_array_module(x)

        axis=self.axis
        if axis is None:
            axis=list(range(len(x.shape)))
        elif not isinstance(axis,(tuple,list)):
            axis=(self.axis,)
        axis=[ax if ax>=0 else len(x.shape)+ax for ax in axis]

        grad_shape=[]
        for ax,size in enumerate(x.shape):
            if ax in axis:
                grad_shape.append(1)
            else:
                grad_shape.append(size)

        max_mask=(xp.reshape(self.outputs[0].data,grad_shape)==x.data).astype(xp.uint8)
        return grad.reshape(grad_shape)*max_mask

# Model Visualization By Graphviz https://zhuanlan.zhihu.com/p/21993254
def plot_graph(output,path):
    dot=Digraph()

    def plot_variable(var):
        dot.node(str(id(var)),var.name if var.name is not None else '',color='gray',style='filled')
    def plot_function(f):
        dot.node(str(id(f)),f.__class__.__name__,color='lightblue',style='filled',shape='box') # function self
        for var in f.inputs: #  input & input to function
            plot_variable(var)
            dot.edge(str(id(var)),str(id(f)))
        for var in f.outputs:   # function to output
            dot.edge(str(id(f)),str(id(var)))

    plot_variable(output)

    func_q=[output.func]
    func_set=set(func_q)
    while len(func_q):
        f=func_q.pop()
        plot_function(f)
        for var in f.inputs:
            if var.func is None or var.func in func_set:
                continue
            func_set.add(var.func)
            func_q.append(var.func)
    dot.render(outfile=path,cleanup=True)

# Do not store inputs&outputs in functions when do inference
NO_GRAD=False
@contextmanager
def no_grad():
    global NO_GRAD
    NO_GRAD=True
    yield
    NO_GRAD=False

if __name__=='__main__':
    print('Add测试')
    x=Variable(2)
    y=Variable(3)
    z=x+y
    print('z:',z)
    z.backward()
    print('x_grad:',x.grad,'y_grad:',y.grad)

    print('Sub测试')
    x=Variable(6)
    y=Variable(4)
    z=x-y
    print('z:',z)
    z.backward()
    print('x_grad:',x.grad,'y_grad:',y.grad)

    print('Mul测试')
    x=Variable(2)
    y=Variable(5)
    z=x*y
    print('z:',z)
    z.backward()
    print('x_grad:',x.grad,'y_grad:',y.grad)

    print('Div测试')
    x=Variable(8)
    y=Variable(2)
    z=x/y
    print('z:',z)
    z.backward()
    print('x_grad:',x.grad,'y_grad:',y.grad)

    print('Pow测试')
    x=Variable(2)
    z=x**4
    print('z:',z)
    #------------ 一阶导数验证 
    z.backward()
    print('x_grad:',x.grad)
    #------------ 二阶导数验证 
    x_grad=x.grad
    x.zero_grad()
    x_grad.backward()
    print('x_double_grad:',x.grad) # y=x^4 -> 4*x^3 -> 12*x^2

    print('Neg测试')
    x=Variable(2)
    z=-x
    print('z:',z)
    #------------ 一阶导数验证 
    z.backward()
    print('x_grad:',x.grad)

    print('复杂算式')
    x=Variable(2)
    y=x*x*x/2+x**2+x+x-x    # y=x^3 / 2 + x^2 + x -> 3*x^2 / 2 + 2*x + 1 -> 6*x/2 + 2
    print('y:',y)
    #------------ 一阶导数验证 
    y.backward()
    print('x_grad:',x.grad)
    #------------ 二阶导数验证 
    x_grad=x.grad
    x.zero_grad()
    x_grad.backward()
    print('x_double_grad:',x.grad) 
    
    print('graphviz可视化')
    x=Variable(3,name='x')
    y=Variable(2,name='y')
    z=x+y+x*y
    z.name='z'
    plot_graph(z,'model.png')

    print('reshape测试')
    x=Variable([1,2,3,4,5,6],name='x')  # shape: (6,)
    y=x.reshape((3,2))  # shape: (3,2)
    y.name='y' 
    print('y:',y)
    y.backward()
    print('x_grad:',x.grad) # shape: (6,)

    print('transpose测试')
    x=Variable([[1,2,3],[4,5,6]],name='x')  # shape: (2,3)
    y=x.transpose()  # shape: (3,2)
    z=y.transpose((1,0))    # 颠倒1和0维度
    y.name='y' 
    z.name='z'
    print('y:',y)
    print('z:',z)
    z.backward()
    print('x_grad:',x.grad) # shape: (2,3)

    print('sum测试')
    x=Variable([[1,2],[3,4]])   # (2,2)
    y=x.sum(axes=(1),keepdims=True) # (2,1)
    y.backward()
    print('z:',y) 
    print('x_grad:',x.grad) # (2,2)

    print('broadcast测试')
    x=Variable(np.arange(1,5).reshape((4,1))) # x shape: (4,1) 
    y=x.broadcast((3,4,2))  # (4,1) -> (3,4,2), 梯度回传累计6倍
    print('y:',y)
    y.backward()    
    print('x_grad:',x.grad) # (4,1)

    print('Add广播兼容性')
    x=Variable([1,2,])
    y=Variable(5)
    z=x+y
    print(z)
    z.backward()
    print('x_grad:',x.grad,'y_grad:',y.grad)

    print('Sub广播兼容性')
    x=Variable([1,2,])
    y=Variable(5)
    z=x-y
    print(z)
    z.backward()
    print('x_grad:',x.grad,'y_grad:',y.grad)

    print('Mul广播兼容性')
    x=Variable([1,2,])
    y=Variable(5)
    z=x*y
    print(z)
    z.backward()
    print('x_grad:',x.grad,'y_grad:',y.grad)

    print('Div广播兼容性')
    x=Variable([4,8,])
    y=Variable(4)
    z=x/y
    print('z:',z)
    z.backward()
    print('x_grad:',x.grad,'y_grad:',y.grad) 

    print('MatMul测试')
    x=Variable(np.random.rand(3,2))
    y=Variable(np.random.rand(2,5))
    z=x@y
    print('z:',z.shape)
    z.backward()
    print('x_grad:',x.grad.shape,'y_grad:',y.grad.shape) 

    print('Slice测试')
    x=Variable([[1,2,3],[4,5,6]])
    x2=x**2
    y=x2[1,1]
    print('y:',y)
    y.backward()
    print('x_grad:',x.grad)
    x_grad=x.grad
    x.zero_grad()
    x_grad.backward()
    print('x_double_grad:',x.grad)
    # plot_graph(x.grad,'slice.png')

    print('Clip测试')
    x=Variable([1,3,5,7,9])
    y=Clip(3,5)(x)
    print('y:',y)
    y.backward()
    print('x_grad:',x.grad)

    print('Max测试')
    x=Variable([
        [1,5,4,5],
        [3,3,4,1]
    ])
    y=Max(axis=(1),keepdims=False)(x)
    y.backward()
    print('max:',y)
    print('x_grad:',x.grad)

    print('线性回归')
    # 准备样本
    np.random.seed(0)
    train_x=np.random.rand(100,1)   #   100个样本x
    train_y=2*train_x+5+np.random.rand(100,1)   # 100个样本y(随机偏离正确y)
    # 定义线性模型
    w=Variable(np.zeros((1,1)))
    b=Variable(np.zeros((1,)))
    lr=0.1
    # 训练
    for i in range(100):
        # forward
        x=Variable(train_x)
        y=x@w+b
        # loss
        loss=((y-train_y)**2).sum()/train_x.shape[0]
        w.zero_grad()
        b.zero_grad()
        # backward
        loss.backward()
        # optimize
        w.data-=lr*w.grad.data
        b.data-=lr*b.grad.data
        print('loss:',loss,'w:',w,'b:',b)

    print('CUDA验证')
    try:
        x=Variable([
            [1,1,1,],
            [2,2,2],
        ]).to_cuda()
        print('[cuda +]',x+2)
        print('[cuda *]',x*2)
        print('[cuda -]',x-2)
        print('[cuda /]',x/2)
        print('[cuda pow]',x**2)
        print('[cuda reshape]',x.reshape((-1,)))
        print('[cuda transpose]',x.T)
        print('[cuda sum]',x.sum())
        print('[cuda broadcast]',x+[1])
        print('[cuda matmul]',x@[[1,1],[1,1],[1,1]])
        print('[cuda slice]',x[:,1])
        print('[cuda log]',Log()(x))
        print('[cuda clip]',Clip(0,1)(x))
        print('[cuda relu]',Relu()(x))
    except Exception as e:
        print('没有NVIDIA显卡,',e)

    print('no_grad验证')
    x1=Variable([1,2,3])
    x2=Variable([4,5,6])
    y=x1+x2
    y.backward()
    print('y=',y,'x1_grad=',x1.grad,'x2_grad=',x2.grad)
    with no_grad():
        y=x1+x2
        try:
            y.backward()
        except Exception as e:
            print('y=',y,'exception=',e)