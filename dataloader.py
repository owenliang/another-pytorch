import math 
import numpy as np 
import cupy as cp
from torch import Variable
from dataset import SpiralDataset

class DataLoader:
    def __init__(self,dataset,batch_size,shuffle=True):
        self.dataset=dataset 
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.batch_count=math.ceil(len(dataset)/batch_size)
        self._reset()

    def _reset(self):
        self.batch_iter=0
        if self.shuffle:
            self.batch_index=np.random.permutation(len(self.dataset))
        else:
            self.batch_index=np.arange(len(self.dataset))

    def __iter__(self):
        return self 
    
    def __next__(self):
        if self.batch_iter==self.batch_count:
            self._reset()
            raise StopIteration()
        x,t=[],[]
        for i in range(self.batch_iter*self.batch_size,(self.batch_iter+1)*self.batch_size):
            x.append(self.dataset[i][0])
            t.append(self.dataset[i][1])
        self.batch_iter+=1
        
        x=Variable(x)
        t=Variable(t)
        return x,t
    
if __name__=='__main__':
    dataset=SpiralDataset()
    dataloader=DataLoader(dataset,30,True)
    for epoch in range(2):
        for batch_i,(batch_x,batch_t) in enumerate(dataloader):
            print('epoch:{} batch_i:{} batch_x:{} batch_t:{}'.format(epoch,batch_i,batch_x,batch_t))