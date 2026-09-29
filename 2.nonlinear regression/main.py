import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
   
test_size=0.2
batch_size=64   
learning_rate=1e-2 
epochs=200 
seed=0


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

X,y=make_friedman1(n_samples=1500,n_features=10,noise=1.0,random_state=42)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size, random_state=42)
X_train=torch.tensor(X_train,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)
X_test=torch.tensor(X_test,dtype=torch.float32)
y_test=torch.tensor(y_test,dtype=torch.float32).unsqueeze(1)
dataset=torch.utils.data.TensorDataset(X_train,y_train)
dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

model1=nn.Linear(in_features=10,out_features=1)

model2=nn.Sequential(
    nn.Linear(in_features=10,out_features=64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,1)
)

print("\n===== model1（线性模型）=====")
print(model1)
print("\n===== model2（MLP）=====")
print(model2)

criterion=nn.MSELoss()
optimizer1=optim.Adam(model1.parameters(),lr=learning_rate)
optimizer2=optim.Adam(model2.parameters(),lr=learning_rate)

hist1={"train_loss": [], "test_loss": []}
hist2={"train_loss": [], "test_loss": []}
def train(model, optimizer, hist):
    """把当前模型在训练集上过一遍，并记录训练/测试 loss。"""
    model.train()
    epoch_loss, batch_num=0.0, 0
    for batch_X, batch_y in dataloader:
        pred=model(batch_X)
        loss=criterion(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()
        batch_num+=1
    model.eval()
    with torch.no_grad():
        test_loss = criterion(model(X_test), y_test).item()

    hist["train_loss"].append(epoch_loss / batch_num)
    hist["test_loss"].append(test_loss)


print("\n%-12s | %-24s | %-24s" % ("Epoch", "线性模型 (train / test)", "MLP (train / test)"))
print("-" * 68)
for epoch in range(epochs):
    train(model1, optimizer1, hist1)      #
    train(model2, optimizer2, hist2)      

    if (epoch+1)%10==0 or epoch==0:
        print("%-12d | %10.4f / %10.4f | %10.4f / %10.4f"
              % (epoch + 1,
                 hist1["train_loss"][-1], hist1["test_loss"][-1],
                 hist2["train_loss"][-1], hist2["test_loss"][-1]))



@torch.no_grad()
def evaluate(model):    
    model.eval()    
    pred=model(X_test)    
    true=y_test    
    mse=criterion(pred,true).item()    
    mae=(pred-true).abs().mean().item()    
    ss_res=((true-pred)**2).sum()    
    ss_tot=((true-true.mean()) ** 2).sum()    
    hist=hist1 if model is model1 else hist2    
    return pred.numpy().ravel(), dict(        
    params=sum(p.numel() for p in model.parameters()),        
    train_loss=hist["train_loss"][-1],        
    MSE=mse,        
    RMSE=mse**0.5,        
    MAE=mae,        
    R2=(1 - ss_res/ss_tot).item(),    
    )

pred1, met1=evaluate(model1)
pred2, met2=evaluate(model2)
print("%-10s %10s %12s %10s %10s %10s %10s" %      
    ("模型", "参数量", "训练末loss", "测试MSE", "测试RMSE", "测试MAE", "测试R2"))
for name, met in [("线性模型", met1), ("MLP", met2)]:    
    print("%-10s %10d %12.4f %10.4f %10.4f %10.4f %10.4f" %          
(name, met["params"], met["train_loss"], met["MSE"], met["RMSE"], met["MAE"], met["R2"]))

plt.rcParams["font.sans-serif"]=["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"]=False
fig, axes = plt.subplots(1,3,figsize=(16, 4.5))
# 图 1：训练/测试 loss 曲线
axes[0].plot(hist1["train_loss"], label="线性-训练", color="#1f77b4")
axes[0].plot(hist1["test_loss"], label="线性-测试", color="#1f77b4", ls="--", alpha=0.7)
axes[0].plot(hist2["train_loss"], label="MLP-训练", color="#ff7f0e")
axes[0].plot(hist2["test_loss"], label="MLP-测试", color="#ff7f0e", ls="--", alpha=0.7)
axes[0].set_xlabel("Training Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("训练 / 测试损失曲线")
axes[0].legend()
axes[0].grid(alpha=0.3)

true = y_test.numpy().ravel()
lim = [true.min(), true.max()]
for ax, title, pred in [(axes[1], "线性模型", pred1), (axes[2], "MLP", pred2)]:    
    ax.scatter(true, pred, s=12, alpha=0.6)    
    ax.plot(lim, lim, "r--", lw=1)    
    ax.set_xlabel("真实值")    
    ax.set_ylabel("预测值")    
    ax.set_title("%s：预测值 vs 真实值" % title)    
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("result.png", dpi=160)