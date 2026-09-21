import torch
import torch.nn as nn

# 1. 基本参数
batch_size = 2
seq_len = 4
embed_dim = 8
num_heads = 2
num_layers = 2

# 2. 构造输入
x = torch.randn(
    batch_size,
    seq_len,
    embed_dim
)

# 3. 构建 Transformer Encoder
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=num_heads,
    batch_first=True
)

encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=num_layers
)

# 4. 前向传播
# TODO
# 将 x 输入 encoder
out=encoder(x)

# 5. 查看结果
print("Input shape:", x.shape)
# TODO
# 输出 Transformer Encoder 的输出 shape
print("Output shape:",out.shape)
# 6. 尝试修改参数
# 修改 num_heads 或 num_layers，再运行一次，观察输出 shape 和模型结构是否发生变化。