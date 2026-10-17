import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=64
learning_rate=0.05
EPOCHS=30
SEED=42
PERTURB_SEED=0
def set_seed(seed):    
    random.seed(seed)    
    np.random.seed(seed)    
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)

# 1. 数据处理
transform = transforms.Compose([
    # TODO
    transforms.Resize([32,32]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
# 噪声
class AddGaussianNoise:
    def __init__(self, sigma=0.1):
        self.sigma = sigma
    def __call__(self, x):
        return x+torch.randn_like(x)*self.sigma
# 训练增强的transform:只作用于训练集
train_aug_transform = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.RandomRotation(10, fill=128),  #小角度旋转
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),  
                            scale=(0.9, 1.1), fill=128),     #平移/缩放
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),  #亮度/对比度
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5,
                                                    sigma=(0.8, 1.6))], p=0.5), #模糊
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),    #部分遮挡
    transforms.RandomApply([AddGaussianNoise(sigma=0.15)], p=0.3),  #噪声
])
#扰动的transform:只作用于测试集
#亮度 / 对比度 / 模糊 / 旋转是PIL级操作，放在ToTensor之前；
#噪声 / 遮挡是张量级操作，必须放在ToTensor之后。
def build_eval_transform(kind):
    ops=[transforms.Resize([32,32])]
    if kind=="bright":
        ops.append(transforms.ColorJitter(brightness=0.6))   #亮度±60%
    elif kind=="contrast":
        ops.append(transforms.ColorJitter(contrast=0.6))     #对比度±60%
    elif kind=="blur":
        ops.append(transforms.GaussianBlur(kernel_size=5,sigma=(0.8,1.6))) #模糊
    elif kind=="rotate":
        ops.append(transforms.RandomRotation(15, fill=128))   #±15°小角度旋转
    elif kind=="occlude":
        pass   #张量集操作
    ops+=[transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
    if kind == "noise":        
        ops.append(AddGaussianNoise(sigma=0.15))    
    elif kind == "occlude":        
        ops.append(transforms.RandomErasing(p=1.0, scale=(0.02, 0.08), value=0))
    return transforms.Compose(ops)

PERTURB_NAMES = {    
"bright": "亮度变化",    
"contrast": "对比度变化",    
"blur": "图像模糊",    
"noise": "高斯噪声",    
"rotate": "小角度旋转",    
"occlude": "部分遮挡",
}

def build_train_loader(augment=False):    
    tf = train_aug_transform if augment else transform    
    ds = torchvision.datasets.GTSRB(root='./data', split="train",download=True, transform=tf)    
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

def build_test_loader(kind=None):      
    tf = transform if kind is None else build_eval_transform(kind)    
    ds = torchvision.datasets.GTSRB(root='./data', split="test",download=True, transform=tf)    
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
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
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
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
        scheduler.step()
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

def train_model(model, criterion, optimizer, train_loader, tag):     
    history= {"train_acc": [], "train_loss": [], "test_acc": [], "test_loss": []}    
    for epoch in range(EPOCHS):        
        model.train()        
        epoch_train_acc, epoch_train_loss = train(train_loader, model, criterion, optimizer)        
        model.eval()        
        epoch_test_acc, epoch_test_loss = test(testloader, model, criterion)        
        history["train_acc"].append(epoch_train_acc)        
        history["train_loss"].append(epoch_train_loss)        
        history["test_acc"].append(epoch_test_acc)        
        history["test_loss"].append(epoch_test_loss)        
        print(f"[{tag}] Epoch:{epoch + 1},Train_acc:{epoch_train_acc * 100:.1f}%,"              
            f"Train_loss:{epoch_train_loss:.4f},Test_acc:{epoch_test_acc * 100:.1f}%,"              
            f"Test_loss:{epoch_test_loss:.4f}")    
    return history

print("\n===== 原CNN =====")
set_seed(SEED)
baseline_hist = train_model(model, criterion, optimizer, trainloader, "baseline")
# 鲁棒性评测
def evaluate_on(model, criterion, kind=None):        
    set_seed(PERTURB_SEED)      # 固定种子        
    return test(build_test_loader(kind), model, criterion)
def evaluate_all(model, criterion, tag):        
    result = {}        
    acc, _ = evaluate_on(model, criterion, None)        
    result["__clean__"] = acc        
    print(f"[{tag}] 原始测试集准确率: {acc * 100:.2f}%")        
    for kind, name in PERTURB_NAMES.items():                
        acc, _ = evaluate_on(model, criterion, kind)                
        result[kind] = acc                
        print(f"[{tag}] {name}准确率: {acc * 100:.2f}%")        
    return result
print("\n===== 模型鲁棒性评测 =====")
base_acc = evaluate_all(model, criterion, "baseline")

# 改进：带数据增强的 transform 重新训练
print("\n=====加入数据增强，重新训练 =====")
set_seed(SEED)
aug_model = SimpleCNN().to(device)
aug_criterion = nn.CrossEntropyLoss()
aug_optimizer = torch.optim.SGD(aug_model.parameters(), lr=learning_rate)
aug_trainloader = build_train_loader(augment=True)
aug_hist = train_model(aug_model, aug_criterion, aug_optimizer, aug_trainloader, "augmented")
aug_acc = evaluate_all(aug_model, aug_criterion, "augmented")

def report(base_acc, aug_acc):    
    header = ["测试条件", "基线准确率(%)", "增强后准确率(%)", "基线下降(%)", "增强后下降(%)"]    
    rows = []    
    for kind in ["__clean__"] + list(PERTURB_NAMES):
        name = "原始测试集" if kind == "__clean__" else PERTURB_NAMES[kind]     
        b, a = base_acc[kind], aug_acc[kind]        
        rows.append([name,                     
            f"{b * 100:.2f}", f"{a * 100:.2f}",                     
            f"{(base_acc['__clean__'] - b) * 100:.2f}",                     
            f"{(aug_acc['__clean__'] - a) * 100:.2f}"])
    print("\n================ 鲁棒性对比结果 ================")    
    print("| " + " | ".join(header) + " |")    
    print("|" + "---|" * len(header))    
    for r in rows:        
        print("| " + " | ".join(r) + " |") 
    worst = max(PERTURB_NAMES, key=lambda k: base_acc["__clean__"] - base_acc[k])    
    print(f"对基线模型影响最大的扰动：{PERTURB_NAMES[worst]}"          
        f"（下降 {(base_acc['__clean__'] - base_acc[worst]) * 100:.2f} 个百分点）")

report(base_acc, aug_acc)


# 5. 可视化
# TODO
def plot_curves(history, title, path):        
    epochs_range = range(1, len(history["train_acc"]) + 1)        
    plt.figure(figsize=(12, 3))        
    plt.subplot(1, 2, 1)        
    plt.plot(epochs_range, history["train_acc"], label="Training Accuracy")        
    plt.plot(epochs_range, history["test_acc"], label="Test Accuracy")        
    plt.legend(loc="lower right")        
    plt.title(title + " - Accuracy")        
    plt.subplot(1, 2, 2)        
    plt.plot(epochs_range, history["train_loss"], label="Training Loss")        
    plt.plot(epochs_range, history["test_loss"], label="Test Loss")        
    plt.legend(loc="upper right")        
    plt.title(title + " - Loss")        
    plt.tight_layout()        
    plt.savefig(path, dpi=150)        
    plt.show()
plot_curves(baseline_hist, "Baseline CNN", os.path.join('result', "fig_baseline_curves.png"))
plot_curves(aug_hist, "CNN + Data Augmentation",            
os.path.join('result', "fig_augmented_curves.png"))