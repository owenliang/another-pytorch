import numpy as np 
from graphviz import Digraph

class Variable:
    def __init__(self,data,name=None):
        self.data=data if isinstance(data,np.ndarray) else np.array(data) # ndarray
        self.name=name  # str
        self.grad=None  # Variable
        self.func=None  # Function
        self.gen=0      # Generation

    def backward(self):
        self.grad=Variable(np.ones_like(self.data))
        
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

    def reshape(self,shape):
        return Reshape(shape)(self)
    
    def transpose(self,axes=None):
        return Transpose(axes)(self)

    def sum(self,axes=None,keepdims=False):
        return Sum(axes,keepdims)(self)

    def broadcast(self,shape):
        return Broadcast(shape)(self)

class Function:
    def __init__(self):
        self.gen=None  # max(inputs' generation)
        self.inputs=None
        self.outputs=None
    
    def forward(self,*inputs): # Variable
        inputs=[var_or_data if isinstance(var_or_data,Variable) else Variable(var_or_data) for var_or_data in inputs] # Variable
        outputs=self._forward(*[var.data for var in inputs])       # ndarray
        if not isinstance(outputs,tuple):
            outputs=outputs,

        # prepare for backward
        self.gen=max([var.gen for var in inputs])
        self.inputs=inputs    # Variable
        self.outputs=[Variable(data) for data in outputs]  # Variable
        for var in self.outputs:
            var.func=self
            var.gen=self.gen+1
        return self.outputs[0] if len(self.outputs)==1 else self.outputs
    
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
        return grad*1,grad*1

# -
class Sub(Function):
    def _forward(self,a,b):
        return a-b 
    
    def _backward(self,grad):
        return grad*1,grad*-1

# *
class Mul(Function):
    def _forward(self,a,b):
        return a*b

    def _backward(self,grad):
        return grad*self.inputs[1],grad*self.inputs[0]    

# /   
class Div(Function):
    def _forward(self,a,b):
        return a/b
    
    def _backward(self,grad):
        return grad*1/self.inputs[1],-1*grad*self.inputs[0]/(self.inputs[1]**2)

# **
class Pow(Function):
    def __init__(self,b):
        self.b=b    # int

    def _forward(self,a):
        return np.power(a,self.b)

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
        self.x_shape=x.shape
        return np.reshape(x,self.output_shape)

    def _backward(self,grad):
        return Reshape(self.x_shape)(grad)

# Matrix Transpose
class Transpose(Function):
    def __init__(self,axes):
        self.axes=axes

    def _forward(self,x):
        return np.transpose(x,self.axes)

    def _backward(self,grad):
        return Transpose(self.axes)(grad)

# Matrix Sum
class Sum(Function):
    def __init__(self,axes,keepdims):
        self.axes=axes
        self.keepdims=keepdims
    
    def _forward(self,x):
        self.x_shape=x.shape
        return np.sum(x,axis=self.axes,keepdims=self.keepdims)

    def _backward(self,grad):   
        # Case1: (3,4,2) -> sum(axes:(0,2),keepdims=True) -> (1,4,1)
        # Case2: (3,4,2) -> sum(axes:(0,2),keepdims=False)-> (4,)
        grad_shape=list(grad.data.shape)
        if len(grad_shape)!=len(self.x_shape):  # keepdims=False
            for idim in self.axes:  # (4,) -> (1,4,) -> (1,4,1)
                grad_shape.insert(idim,1)
        
        grad=grad.reshape(grad_shape)  # Reshape function  ,    (1,4,1)
        grad=grad.broadcast(self.x_shape)   # Broadcast function    (1,4,1)->(3,4,2)
        return grad

# Matrix DeBroadcast
class DeBroadcast(Function):
    def __init__(self,output_shape):
        self.output_shape=output_shape

    def _forward(self,x):   # x:(3,4,2) , output_shape: (4,1)
        self.x_shape=x.shape

        prefix_ndim=len(x.shape)-len(self.output_shape)  # len((3,4,2))-len((4,1))
        dims=[]
        for idim in range(len(self.output_shape)):
            if self.output_shape[idim]!=x.shape[prefix_ndim+idim]:
                dims.append(prefix_ndim+idim)
        prefix_dims=list(range(prefix_ndim))
        output=np.sum(x,axis=tuple(prefix_dims+dims),keepdims=True)
        return np.squeeze(output,axis=tuple(prefix_dims))

    def _backward(self,grad):   # grad:(4,1), return: (3,4,2)
        return Broadcast(self.x_shape)(grad)

# Matrix Broadcast
class Broadcast(Function):
    def __init__(self,output_shape):
        self.output_shape=output_shape 
    
    def _forward(self,x):   
        self.x_shape=x.shape
        # Case1: Simple version,  (1,4,1)   -> (3,4,2)
        # Case2: Hard version, (4,1)  ->    (3,4,2)
        return np.broadcast_to(x,self.output_shape)

    def _backward(self,grad):    # Case2: Hard version,   grad: (3,4,2)  -> (4,1)
        return DeBroadcast(self.x_shape)(grad)


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

if __name__=='__main__':
    print('加法测试')
    x=Variable(2)
    y=Variable(3)
    z=x+y
    print('z:',z)
    z.backward()
    print('x_grad:',x.grad,'y_grad:',y.grad)

    print('减法测试')
    x=Variable(6)
    y=Variable(4)
    z=x-y
    print('z:',z)
    z.backward()
    print('x_grad:',x.grad,'y_grad:',y.grad)

    print('乘法测试')
    x=Variable(2)
    y=Variable(5)
    z=x*y
    print('z:',z)
    z.backward()
    print('x_grad:',x.grad,'y_grad:',y.grad)

    print('除法测试')
    x=Variable(8)
    y=Variable(2)
    z=x/y
    print('z:',z)
    z.backward()
    print('x_grad:',x.grad,'y_grad:',y.grad)

    print('幂测试')
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

    print('取反测试')
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