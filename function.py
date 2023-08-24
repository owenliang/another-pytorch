from variable import Variable
import numpy as np 

class Function:
    def __init__(self):
        self.gen=None  # 赋值为max(inputs.gen)
        self.inputs=None
        self.outputs=None
    
    # 前向传播
    def forward(self,*inputs): # Variable
        inputs=[var_or_data if isinstance(var_or_data,Variable) else Variable(var_or_data) for var_or_data in inputs] # 输入需要强转成Variable
        outputs=self._forward(*[var.data for var in inputs])       # ndarray
        if not isinstance(outputs,tuple):
            outputs=outputs,

        # 反向传播需要的
        self.gen=max([var.gen for var in inputs])
        self.inputs=inputs    # Variable
        self.outputs=[Variable(data) for data in outputs]  # Variable
        for var in self.outputs:
            var.func=self
            var.gen=self.gen+1
        return self.outputs[0] if len(self.outputs)==1 else self.outputs
    
    __call__=forward

    # 反向传播
    def backward(self):
        output_grads=[var.grad for var in self.outputs]    # Variable
        input_grads=self._backward(*output_grads)   # Variable
        for i in range(len(self.inputs)):
            if self.inputs[i].grad is None:
                self.inputs[i].grad=input_grads[i]
            else:
                self.inputs[i].grad=Add()(self.inputs[i].grad,input_grads[i])  # Variable+Variable
        # 清理不用的梯度
        for var in self.outputs:
           var.grad=None

    # 具体实现
    def _forward(self,*inputs):    # ndarray
        raise NotImplementedError()
    
    def _backward(self,*grad):   # Variable
        raise NotImplementedError()
    
class Mul(Function):
    def _forward(self,a,b):
        return a*b

    def _backward(self,grad):
        return Mul()(self.inputs[1],grad),Mul()(self.inputs[0],grad)

class Add(Function):
    def _forward(self,a,b):
        return a+b

    def _backward(self,grad):
        return Mul()(1,grad),Mul()(1,grad)
    
if __name__=='__main__':
    from variable import Variable

    # (x+y)*x
    
    #------------ 一阶导数验证 
    x=Variable(1)
    y=Variable(2)
    add=Add()
    sum=add(x,y)
    mul=Mul()
    final=mul(sum,x)
    
    final.backward()
    print('x_grad:',x.grad,'y_grad:',y.grad)     # 2x+y=4,    x=1

    #------------ 二阶导数验证
    gx=x.grad
    x.zero_grad()
    y.zero_grad()
    gx.backward()
    print('x_grad2:',x.grad,'y_grad2:',y.grad)  # 2,    1