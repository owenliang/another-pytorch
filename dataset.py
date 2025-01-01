import numpy as np 
from layers import *
from utils import get_file
import gzip

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

def get_spiral(train=True):
    seed=1984 if train else 2020
    np.random.seed(seed=seed)

    num_data,num_class,input_dim=100,3,2
    data_size=num_class*num_data 
    x=np.zeros((data_size,input_dim),dtype=np.float32)
    t=np.zeros(data_size,dtype=np.int32)

    for j in range(num_class):
        for i in range(num_data):
            rate=i/num_data
            radius=1.0*rate
            theta=j*4.0+4.0*rate+np.random.randn()*0.2
            ix=num_data*j+i
            x[ix]=np.array([radius*np.sin(theta),radius*np.cos(theta)]).flatten() 
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

class MNISTDataset(Dataset):
    def __init__(self,train=True,transformer=None,target_transformer=None):
        super().__init__(transformer,target_transformer)

        url='http://yann.lecun.com/exdb/mnist/'
        train_files={
            'data': 'train-images-idx3-ubyte.gz',
            'label': 'train-labels-idx1-ubyte.gz'
        }
        test_files={
            'data': 't10k-images-idx3-ubyte.gz',
            'label': 't10k-labels-idx1-ubyte.gz'
        }
    
        files=train_files if train else test_files
        data_path=get_file(url+files['data'])
        label_path=get_file(url+files['label'])

        self.data=self._load_data(data_path)
        self.label=self._load_label(label_path)
    
    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data=np.frombuffer(f.read(),np.uint8,offset=16)
        data=data.reshape(-1, 1, 28, 28)
        return data
    
    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels=np.frombuffer(f.read(),np.uint8,offset=8)
        return labels
    
    def _len(self):
        return len(self.data)

    def _getitem(self,index):
        return self.data[index],self.label[index]

class SinDataset(Dataset):
    def __init__(self,transformer=None,target_transformer=None):
        super().__init__(transformer,target_transformer)
        self.x=np.linspace(0,4*2*np.pi,100) # 5 cycles, 2pi per cycle, 10000 sample points 
        self.y=np.sin(self.x)
        
        steps=50
        self.steps=steps
        self.samples=[]
        for i in range(0,len(self.x)-steps,steps):
            y=self.y[i:i+steps]
            x=y[:-1]
            t=y[1:]
            self.samples.append((x,t))
    
    def _len(self):
        return len(self.samples)

    def _getitem(self,index):
        return self.samples[index]

if __name__=='__main__':    
    import matplotlib.pyplot as plt
    
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

    print('MNISTDataset')
    mnist_dataset=MNISTDataset()
    x,t=mnist_dataset[0]
    plt.imshow(x.reshape(28,28))
    plt.show()
    
    print('SinDataset')
    sin_dataset=SinDataset()
    plt.plot(sin_dataset.x,sin_dataset.y)
    plt.show()