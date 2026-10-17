# MiniMind-3 复现实验


1. 环境与硬件

| 项目 | 内容 |
| --- | --- |
| 操作系统 | Windows |
| GPU | NVIDIA GeForce RTX 5060 Laptop，8GB 显存，驱动 582.05 |
| Python | 3.11.16（conda环境 `minimind`） |
| PyTorch | 2.11.0+cu128（CUDA 12.8） |
| Transformers | 4.57.6 |
| 其它关键依赖 | datasets 3.6.0 / numpy 1.26.4 / pyarrow 25.0.1 / scikit-learn 1.5.1 |



2. 运行命令

```bash
# 环境
conda activate minimind
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 下载模型
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3

# 基础推理
python eval_llm.py --load_from ./minimind-3 --max_new_tokens 256 --device cuda

# 训练
cd trainer
python train_pretrain.py                          # 产出 ../out/pretrain_768.pth
python train_full_sft.py                          # 产出 ../out/full_sft_768.pth


# 对比两个阶段的模型
cd ..
python eval_llm.py --load_from ./model --weight pretrain 
python eval_llm.py --load_from ./model --weight full_sft 
```

3. 训练结果

| 阶段 | epoch | 步数/epoch | 起始 loss | 结束 loss | 学习率 | 平均步时 | 两轮耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 预训练 | 2 | 39,695 | 7.5217 | 1.6469 | 5e-4 → 5e-5 | ≈0.39 s/步 | ≈8.7 小时 |
| SFT | 2 | 56,608 | 2.1057 | 1.7119 | 1e-5 → 1e-6 | ≈0.44 s/步 | ≈13.9 小时 |


**checkpoint **

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `out/pretrain_768.pth` | 131.3 MB | 预训练最终权重 |
| `out/full_sft_768.pth` | 131.3 MB | SFT 最终权重 |
| `checkpoints/*_resume.pth` | 619 MB | 含优化器状态，用于断点续训 |





4. 复现过程中遇到的问题及解决


- 问题 1：`import datasets` 报 DLL 加载失
现象：`ImportError: DLL load failed while importing _compute: 应用程序控制策略已阻止此文件`。
原因：pyarrow 的 `_compute.pyd` 被应用控制策略拦截。
解决：换用 pyarrow 25.0.1（同时满足 `datasets 3.6.0` 要求的 `pyarrow>=15`）。

- 问题 2：`import torch` 报 WinError 1114
现象：`OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败 ... torch\lib\c10.dll`。
原因：Windows 下 pyarrow 与 torch 的 DLL 冲突）。
解决：把 pyarrow 升级到 25.0.1 后冲突消失。

- 问题 3：笔记本 GPU 训练慢
现象：8 GB 显存的笔记本 GPU 上，SFT 单 epoch 约 7 小时（README 在 3090 上的估算是 1.1 小时）。
原因：显存占用已达 7290/8151 MiB，GPU 利用率 100%、功耗受 65 W 限制。
解决：不改配置、接受更长训练时间；如要加速可减小 batch 并用梯度累积，或改用云端 GPU。

