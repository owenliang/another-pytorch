import numpy as np 

class Variable:
    def __init__(self,data):
        self.data=data if isinstance(data,np.ndarray) else np.array(data)  # 强制转换ndarray
        self.grad=None  # Variable
        self.func=None  # Function
        self.gen=0      # 赋值为算子gen+1

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
            func_q=sorted(func_q,key=lambda f:f.gen,reverse=True)

    def zero_grad(self):
        self.grad=None

    def __repr__(self):
        return str(self.data)