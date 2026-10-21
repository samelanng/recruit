# MiniMind-3 复现实验


1. 环境与硬件

| 项目 | 内容 |
| --- | --- |
| 操作系统 | Windows |
| GPU | True NVIDIA GeForce RTX 3090 内存24G |
| Python | 3.12 |
| PyTorch | 2.5.1+cu124（CUDA 12.4） |
| Transformers | 4.57.6 |
| 其它关键依赖 | datasets 3.6.0 / numpy 1.26.4 / pyarrow 17.0.0 |



2. 运行命令

```bash
# 环境
apt-get update -qq && apt-get install -y tmux
source /etc/network_turbo
cd /root/autodl-tmp
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind
# 修复 Python 3.12 不兼容的 ujson
python -c "p='requirements.txt'; s=open(p,encoding='utf-8').read().replace('ujson==5.1.0','ujson==5.13.0'); open(p,'w',encoding='utf-8').write(s)"
grep ujson requirements.txt
# 装依赖 + 固定 pyarrow
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
pip install "pyarrow==17.0.0" -i https://mirrors.aliyun.com/pypi/simple

# 下载数据
pip install modelscope -i https://mirrors.aliyun.com/pypi/simple

cd /root/autodl-tmp/minimind
mkdir -p dataset

modelscope download --dataset gongjy/minimind_dataset pretrain_t2t_mini.jsonl --local_dir ./dataset
modelscope download --dataset gongjy/minimind_dataset sft_t2t_mini.jsonl --local_dir ./dataset


#预训练
cd /root/autodl-tmp/minimind
mkdir -p logs

tmux new -s train
cd trainer
python -u train_pretrain.py 2>&1 | tee ../logs/pretrain.log

# SFT
cd /root/autodl-tmp/minimind/trainer
python -u train_full_sft.py 2>&1 | tee ../logs/sft.log


#对比两个阶段的模型
cat > /tmp/questions.txt <<'EOF'
1
请用不超过 100 字解释什么是机器学习。
用 Python 编写一个判断质数的函数。
一名学生连续三天复习，第一天 2 小时，第二天 3 小时，第三天比前两天总和少 1 小时。第三天复习了多久？
将下面这句话改写得更正式一些：老师你好，我今天有事，作业能不能晚点交？
请为"人工智能社团招新"活动写一句简短宣传语。

EOF

cd /root/autodl-tmp/minimind

python -u eval_llm.py --weight pretrain  --max_new_tokens 200 < /tmp/questions.txt > logs/eval_pretrain.log 2>&1
python -u eval_llm.py --weight full_sft --max_new_tokens 200 < /tmp/questions.txt > logs/eval_full_sft.log 2>&1

cat logs/eval_pretrain.log
cat logs/eval_full_sft.log

#保存日志、曲线、checkpoint、推理结果
mkdir -p /root/autodl-tmp/results

cp -r /root/autodl-tmp/minimind/logs       /root/autodl-tmp/results/
cp /root/autodl-tmp/minimind/out/pretrain_768.pth /root/autodl-tmp/results/
cp /root/autodl-tmp/minimind/out/full_sft_768.pth /root/autodl-tmp/results/
cp -r /root/autodl-tmp/minimind/checkpoints /root/autodl-tmp/results/

#绘图loss曲线
python - <<'EOF'
import re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pat = re.compile(r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), loss: ([\d.]+)")

for name in ["pretrain", "sft"]:
    xs, ys = [], []
    for line in open(f"/root/autodl-tmp/results/logs/{name}.log", encoding="utf-8", errors="ignore"):
        m = pat.search(line)
        if m:
            ep, _, step, total, loss = m.groups()
            xs.append((int(ep) - 1) * int(total) + int(step))
            ys.append(float(loss))
    if xs:
        plt.figure(figsize=(10, 5))
        plt.plot(xs, ys, lw=1.2)
        plt.yscale("log")
        plt.xlabel("Training Step"); plt.ylabel("Loss"); plt.title(name)
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"/root/autodl-tmp/results/loss_{name}.png", dpi=160)
        print(name, len(xs), "points")
EOF

ls -lh /root/autodl-tmp/results/
```

3. 结果
见`results`文件夹



4. 复现过程中遇到的问题及解决

（本地训练）
- 问题 1：`import datasets` 报 DLL 加载失
现象：`ImportError: DLL load failed while importing _compute: 应用程序控制策略已阻止此文件`。
原因：pyarrow 的 `_compute.pyd` 被应用控制策略拦截。
解决：换用 pyarrow 25.0.1（同时满足 `datasets 3.6.0` 要求的 `pyarrow>=15`）。

- 问题 2：`import torch` 报 WinError 1114
现象：`OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败 ... torch\lib\c10.dll`。
原因：Windows 下 pyarrow 与 torch 的 DLL 冲突）。
解决：把 pyarrow 升级到 25.0.1 后冲突消失。

- 问题 3：笔记本 GPU 训练慢
现象：8 GB 显存的笔记本 GPU 上，SFT 单 epoch 约 7 小时。
原因：显存占用已达 7290/8151 MiB，GPU 利用率 100%、功耗受 65 W 限制。
解决：不改配置、接受更长训练时间；如要加速可减小 batch 并用梯度累积，或改用云端 GPU。

（云端训练）
- 问题1：parrow、numpy版本冲突
解决：把把 pyarrow 降到兼容 NumPy 1.x 的版本（pyarrow 17.0.0）。

