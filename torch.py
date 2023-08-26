import numpy as np 

class Variable:
    def __init__(self,data):
        self.data=data if isinstance(data,np.ndarray) else np.array(data) # ndarray
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

    def __mul__(self,other):
        return Mul()(self,other)

    def __rmul__(self,other):
        return Mul()(other,self)

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
        input_grads=self._backward(*output_grads)   # Variable
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
    
class Mul(Function):
    def _forward(self,a,b):
        return a*b

    def _backward(self,grad):
        return self.inputs[1]*grad,self.inputs[0]*grad

class Add(Function):
    def _forward(self,a,b):
        return a+b

    def _backward(self,grad):
        return 1*grad,1*grad
    
if __name__=='__main__':
    #------------ 一阶导数验证 
    x=Variable(2)
    y=x*x*x
    y.backward()
    print('x_grad:',x.grad)

    #------------ 二阶导数验证 
    x_grad=x.grad
    x.zero_grad()
    x_grad.backward()
    print('x_double_grad:',x.grad) # y=x^3 -> y=3*x^2 -> y=6*x