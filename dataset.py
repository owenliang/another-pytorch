import numpy as np 
from layers import *

class Dataset:
    def __init__(self,transformer=None,target_transformer=None):
        self.transformer=transformer
        self.target_transformer=target_transformer

    def __len__(self):
        return self._len()
    
    def __getitem__(self,index):
        x,t=self._getitem(index)
        if x is not None and self.transformer is not None:
            x=self.transformer(x)
        if t is not None and self.target_transformer is not None:
            t=self.target_transformer(t)
        return x,t
    
    # Overwrite
    def _getitem(self,index):
        raise NotImplementedError()
    
    # Overwrite
    def _len(self):
        raise NotImplementedError()

# 漩涡数据集
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

class SpiralDataset(Dataset):
    def __init__(self,train=True,transformer=None,target_transformer=None):
        super().__init__(transformer,target_transformer)
        self.x,self.t=get_spiral(train)
    
    def _len(self):
        return len(self.x)

    def _getitem(self,index):
        return self.x[index],self.t[index]

if __name__=='__main__':
    def transformer(x):
        print('transformer x:',x)
        return x 
    def target_transformer(t):
        print('target_transformer t:',t)
        return t
    spiral_dataset=SpiralDataset(train=True,transformer=transformer,target_transformer=target_transformer)
    print('SpiralDataset size:',len(spiral_dataset))
    for i in np.random.permutation(len(spiral_dataset))[:5]:
        print('spiral_dataset[{}] x:{} t:{}'.format(i,spiral_dataset[i][0],spiral_dataset[i][1]))