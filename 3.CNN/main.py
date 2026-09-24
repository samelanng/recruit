import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=64
learning_rate=1e-4
EPOCHS=100
# 1. 数据处理
transform = transforms.Compose([
    # TODO
    transforms.Resize([32,32]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

trainset = torchvision.datasets.GTSRB(root="./data", split="train", download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.GTSRB(root="./data", split="test", download=True, transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# 2. 构建 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
        # 定义网络层
        self.main=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=0),#[12,30,30]
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12,12,3,1,0),#[12,28,28]
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2,2),#[12,14,14]
            nn.Conv2d(12,24,3,1,0),#[24,12,12]
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24,24,3,1,0),#[24,10,10]
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2,2),#[24,5,5]
            nn.Flatten(),
            nn.Linear(24*5*5,43)
        )


    def forward(self, x):
        # TODO
        # 定义前向传播逻辑
        x=self.main(x)
        return x

model = SimpleCNN().to(device)

# 3. 损失函数与优化器
# TODO
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
# 4. 训练与评估
# TODO
def train(dataloader,model,criterion,optimizer):
    size=len(dataloader.dataset)
    batch_num=len(dataloader)
    train_loss,train_acc=0,0
    for X,y in dataloader:
        X,y=X.to(device),y.to(device)
        y_pred=model(X)
        loss=criterion(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc+=(y_pred.argmax(1)==y).type(torch.float).sum().item()
        train_loss+=loss.item()
    train_acc/=size
    train_loss/=batch_num
    return train_acc,train_loss

def test(dataloader,model,criterion):
    size=len(dataloader.dataset)
    batch_num=len(dataloader)
    test_loss,test_acc=0,0
    with torch.no_grad():
        for imgs,target in dataloader:
            imgs,target=imgs.to(device),target.to(device)
            target_pred=model(imgs)
            loss=criterion(target_pred,target)
            test_acc+=(target_pred.argmax(1)==target).type(torch.float).sum().item()
            test_loss+=loss.item()
    test_acc/=size
    test_loss/=batch_num
    return test_acc,test_loss
train_loss=[]
train_acc=[]
test_loss=[]
test_acc=[]
for epoch in range(EPOCHS):
    model.train()
    epoch_train_acc,epoch_train_loss=train(trainloader,model,criterion,optimizer)
    model.eval()
    epoch_test_acc,epoch_test_loss=test(testloader,model,criterion)
    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)

    print(f"Epoch:{epoch+1},Train_acc:{epoch_train_acc*100:.1f}%,Train_loss:{epoch_train_loss:.4f},Test_acc:{epoch_test_acc*100:.1f}%,Test_loss:{epoch_test_loss:.4f}")


# 5. 可视化
# TODO
epochs_range=range(EPOCHS)
plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.plot(epochs_range,train_acc,label='Training Accuracy')
plt.plot(epochs_range,test_acc,label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range,train_loss,label='Training Loss')
plt.plot(epochs_range,test_loss,label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()