from sklearn.datasets import make_friedman1
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

batch_size=10
learning_rate=1e-3
epochs=100

X,y=make_friedman1(n_samples=1500,n_features=10,noise=1.0,random_state=42)
X=torch.tensor(X,dtype=torch.float32)
y=torch.tensor(y,dtype=torch.float32).unsqueeze(1)
dataset=torch.utils.data.TensorDataset(X,y)
dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

model1=nn.Linear(in_features=10,out_features=1)

model2=nn.Sequential(
    nn.Linear(in_features=10,out_features=64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,1)
)



criterion=nn.MSELoss()
optimizer1=optim.SGD(model1.parameters(),lr=learning_rate)
optimizer2=optim.Adam(model2.parameters(),lr=learning_rate)

train_loss1=[]
train_loss2=[]
for epoch in range(epochs):
    epoch_loss1,epoch_loss2=0,0
    batch_num=0
    for batch_X,batch_y in dataloader:
        y_pred=model1(batch_X)
        loss=criterion(y_pred,batch_y)
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        epoch_loss1+=loss.item()
        batch_num+=1
    avg_epoch_loss1=epoch_loss1/batch_num
    train_loss1.append(avg_epoch_loss1)

    for batch_X,batch_y in dataloader:
        mlp_pred=model2(batch_X)
        mlp_loss=criterion(mlp_pred,batch_y)
        optimizer2.zero_grad()
        mlp_loss.backward()
        optimizer2.step()
        epoch_loss2+=mlp_loss.item()
    avg_epoch_loss2=epoch_loss2/batch_num
    train_loss2.append(avg_epoch_loss2)
    
    if(epoch+1)%10==0:
        print(f"Epoch:{epoch+1},Linear Loss:{avg_epoch_loss1:.4f},MLP Loss:{avg_epoch_loss2:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(train_loss1, label="Linear Model", color="#1f77b4")
plt.plot(train_loss2, label="MLP Model", color="#ff7f0e")
plt.xlabel("Training Epoch")
plt.ylabel("Average Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
