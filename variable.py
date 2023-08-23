import numpy as np 

class Variable:
    def __init__(self,data):
        self.data=data if isinstance(data,np.ndarray) else np.array(data)  # 强制转换ndarray
        self.grad=None  # Variable
        self.func=None  # Function
        self.gen=0      # 等于算子gen+1,backward逐层反传

    def __repr__(self):
        return 'Variable[gen:{}\tdata:{}\tgrad:{}\tfunc:{}]'.format(self.gen,self.data,self.grad,self.func)

if __name__=='__main__':
    import numpy as np
    var=Variable(2.0)
    print(var)