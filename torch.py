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