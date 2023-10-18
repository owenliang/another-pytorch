import numpy as np 
from layers import *

# 人工制造的分类数据集(螺旋分布)
def get_spiral(train=True):
    seed=1984 if train else 2020
    np.random.seed(seed=seed)

    num_data,num_class,input_dim=100,3,2
    data_size=num_class*num_data    # 3个分类,每个分类100个样本
    x=np.zeros((data_size,input_dim),dtype=np.float32)
    t=np.zeros(data_size,dtype=np.int32)

    for j in range(num_class):
        for i in range(num_data):
            rate=i/num_data
            radius=1.0*rate
            theta=j*4.0+4.0*rate+np.random.randn()*0.2
            ix=num_data*j+i
            x[ix]=np.array([radius*np.sin(theta),radius*np.cos(theta)]).flatten() # 2个特征分别按sin,cos分布
            t[ix]=j
    # Shuffle
    indices=np.random.permutation(num_data * num_class)
    x=x[indices]
    t=t[indices]
    return x,t

if __name__=='__main__':
    pass