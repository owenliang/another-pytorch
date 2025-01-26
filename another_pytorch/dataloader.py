import math
import numpy as np
from another_pytorch.torch import Variable
from another_pytorch.dataset import SpiralDataset

class Sampler:
    def __iter__(self):
        raise NotImplementedError()

class SequentialSampler(Sampler):
    def __init__(self,dataset):
        self.dataset=dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))
    
class RandomSampler(Sampler):
    def __init__(self,dataset,seed=0):
        self.dataset=dataset
        self.rng=np.random.default_rng(seed)

    def __iter__(self):
        return iter(self.rng.permutation(len(self.dataset)))
    
class DefaultCollateFn:
    def __call__(self,batch):
        tensors=[[] for _ in range(len(batch[0]))]
        for sample in batch:
            for i in range(len(sample)):
                tensors[i].append(sample[i])
        return tuple([Variable(t) for t in tensors])
        
class DataLoader:
    def __init__(self,dataset,batch_size,shuffle=False,sampler=None,collate_fn=None):
        self.dataset=dataset 
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.sampler=sampler
        if self.sampler is None:
            self.sampler=RandomSampler(self.dataset) if self.shuffle else SequentialSampler(self.dataset)
        self.collate_fn=collate_fn
        if self.collate_fn is None:
            self.collate_fn=DefaultCollateFn()

    def __iter__(self):
        batch_count=math.ceil(len(self.dataset)/self.batch_size)
        sampler_iter=iter(self.sampler)
        for _ in range(batch_count):
            batch=[]
            for _ in range(self.batch_size):
                try:
                    sample=self.dataset[next(sampler_iter)]
                    batch.append(sample)
                except StopIteration:
                    break
            yield self.collate_fn(batch)
    
if __name__=='__main__':
    dataset=SpiralDataset()
    dataloader=DataLoader(dataset,batch_size=280,shuffle=True)
    for epoch in range(2):
        for batch_i,(batch_x,batch_t) in enumerate(dataloader):
            print('epoch:{} batch_i:{} batch_x:{} batch_t:{}'.format(epoch,batch_i,batch_x,batch_t))