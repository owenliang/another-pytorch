from another_pytorch.layers import *
from another_pytorch.optimizers import * 
from another_pytorch.dataloader import *
from another_pytorch.dataset import * 

print('[CUDA]MNIST分类模型')

# model
class MNIST_MLP(Layer):
    def __init__(self):
        super().__init__()
        self.linear1=Linear(28*28,1000)
        self.relu1=Relu()
        self.linear2=Linear(1000,10)
    
    def _forward(self,x): 
        y=self.linear1(x)
        y=self.relu1(y)
        return self.linear2(y)

# feature transofmer
def img_transformer(x):
    x=x.flatten()
    x=x/256.0
    return x 

# evaluation
def accuracy(output,t):
    pred_t=output.data.argmax(axis=-1)
    acc=(pred_t==t.data).sum()/t.shape[0]
    return Variable(acc)

# hyper parameter
epoch=20
batch_size=1000

# dataset
train_dataset=MNISTDataset(train=True,transformer=img_transformer)
test_dataset=MNISTDataset(train=False,transformer=img_transformer)

# dataloader
train_dataloader=DataLoader(train_dataset,batch_size)
test_dataloader=DataLoader(test_dataset,batch_size)

# model
model=MNIST_MLP().to_cuda()

# optimizer
optimizer=MomentumSGB(model.params(),lr=0.1)

# loss function
loss_fn=SoftmaxCrossEntropy1D().to_cuda()

# training
try:
    for e in range(epoch):
        epoch_loss=0
        epoch_acc=0
        iters=0
        for x,t in train_dataloader:
            x=x.to_cuda()
            t=t.to_cuda() 
            output=model(x)

            loss=loss_fn(output,t)
            model.zero_grads()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.data
            iters+=1
            acc=accuracy(output,t)
            epoch_acc+=acc
        print('avg_loss:',epoch_loss/iters,'avg_acc:',epoch_acc/iters)
except Exception as e:
    print('没有NVIDIA显卡,',e)