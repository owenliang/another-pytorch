import numpy as np
from another_pytorch.torch import Variable

class Sampler:
    def __iter__(self):
        raise NotImplementedError()
    
    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self,state_dict):
        raise NotImplementedError()

class SequentialSampler(Sampler):
    def __init__(self,dataset):
        self.dataset=dataset
        self.idx=0

    def __iter__(self):
        while self.idx<len(self.dataset):
            self.idx+=1
            yield self.idx-1
        self.idx=0
        
    def state_dict(self):
        return {'idx':self.idx}

    def load_state_dict(self,state_dict):
        self.idx=state_dict['idx']
    
class RandomSampler(Sampler):
    def __init__(self,dataset,seed=0):
        self.dataset=dataset
        self.rng=np.random.default_rng(seed)
        self.idx=0
        self.indexes=None
        self.rng_state=self.rng.bit_generator.state

    def __iter__(self):
        if self.indexes is None:
            self.indexes=self.rng.permutation(len(self.dataset))
        while self.idx<len(self.dataset):
            self.idx+=1
            yield self.indexes[self.idx-1]
        self.idx=0
        self.indexes=None
        self.rng_state=self.rng.bit_generator.state
    
    def state_dict(self):
        return {'idx':self.idx,'rng_state':self.rng_state}
    
    def load_state_dict(self,state_dict):
        self.idx=state_dict['idx']
        self.rng_state=state_dict['rng_state']
        self.rng.bit_generator.state=self.rng_state
    
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
        sampler_iter=iter(self.sampler)
        while True:
            batch=[]
            iter_end=False
            for _ in range(self.batch_size):
                try:
                    sample=self.dataset[next(sampler_iter)]
                    batch.append(sample)
                except StopIteration:
                    iter_end=True
                    break
            if len(batch):
                yield self.collate_fn(batch)
            if iter_end:
                break
    
    def state_dict(self):
        return {'batch_size':self.batch_size,'sampler':self.sampler.state_dict()}
    
    def load_state_dict(self,state_dict):
        self.batch_size=state_dict['batch_size']
        self.sampler.load_state_dict(state_dict['sampler'])
    
if __name__=='__main__':
    from another_pytorch.dataset import SpiralDataset,MNISTDataset
    
    print('DataLoader迭代')
    dataset=SpiralDataset()
    dataloader=DataLoader(dataset,batch_size=280,shuffle=True)
    for epoch in range(2):
        for batch_i,(batch_x,batch_t) in enumerate(dataloader):
            print('epoch:{} batch_i:{} batch_x:{} batch_t:{}'.format(epoch,batch_i,batch_x.shape,batch_t.shape))
            
    print('DataLoader状态保存和加载')
    dataset=MNISTDataset()
    dataloader=DataLoader(dataset,batch_size=10,shuffle=True)
    total_samples=0
    for batch_i,(batch_x,batch_y) in enumerate(dataloader):
        total_samples+=batch_x.shape[0]
        if batch_i == 100:
            dataloader_state=dataloader.state_dict()
            print('checkpoint at total_samples:{}'.format(total_samples))
            break
    
    dataloader=DataLoader(dataset,batch_size=10,shuffle=True)
    dataloader.load_state_dict(dataloader_state)
    for batch_i,(batch_x,batch_y) in enumerate(dataloader):
        total_samples+=batch_x.shape[0]
    print('all finished total_samples:{}, dataset:{}'.format(total_samples,len(dataset)))
    
    dataloader_state=dataloader.state_dict()
    dataloader_restore=DataLoader(dataset,batch_size=10,shuffle=True)
    dataloader_restore.load_state_dict(dataloader_state)
    dataloader_iter=iter(dataloader)
    dataloader_restore_iter=iter(dataloader_restore)
    match=True
    for i in range(100):
        batch_x,batch_y=next(dataloader_iter)
        batch_x_restore,batch_y_restore=next(dataloader_restore_iter)
        match=match and np.allclose(batch_x.data,batch_x_restore.data) and np.allclose(batch_y.data,batch_y_restore.data)
    print('DataLoader epoch断点状态恢复正确:{}'.format(match))