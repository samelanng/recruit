# Transformer

## 理论问答

- 1.在 Transformer 出现之前，RNN、LSTM 等序列模型主要存在哪些局限？

RNN等序列模型是时序神经网络，难以并行计算，时间消耗长，计算性能差；容易丢失早期的信息；若要通过增大$h_t$来降低丢失早期信息的可能性，又会增大内存消耗。

- 2.Transformer 与 RNN 在处理序列时，最大的结构区别是什么？
不使用循环层和卷积，只依赖注意力机制
- 3.Transformer 中最核心的公式之一是：​
  
$$
\text{Attention}(Q,K,V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}}\right)V
$$
​a.$Q$、$V$、$T$分别起到什么作用？
$Q$ $V$ $T$
b.$QK^T$得到的结果是什么？
相似度 权重
c.softmax在这里做了什么？
非负，和为1
d.为什么要除以$\sqrt{d_k}$？如果不缩放，可能带来什么问题？
若$d_k$比较大的时候，点积运算的结果也会较大（小），各结果之间的差距也会增大，通过softmax后大的值会更加靠近1，另外的值会更加靠近0，向两端靠拢，导致计算梯度时梯度较小

- 4.为什么要使用多个 Attention Head？
多输出通道，识别不同的模式，增加学习参数
- 5.为什么 Transformer 需要 Positional Encoding？为什么作者使用$sin$和$cos$函数来表示位置信息？
在输入中加入时序信息​
- 6.为什么 Decoder 中需要使用 Masked Self-Attention？​
避免看到$t$时刻以后的输入，保证训练和预测的一致
- 7.Decoder 和 Encoder 相比，多了什么 Attention 结构？​
masked
- 8.Residual Connection 和 Layer Normalization 在 Transformer 中分别起什么作用？

tip 残差连接
encoder:将长为$n$的序列$[x_1,x_2,x_3...x_n]$表示为$[z_1,z_2,z_3...z_n]$，其中的$z_t$是$x_t$的向量表示

multi-head self-attention
全连接层(MLP)
LayerNorm：对样本归一化(垂直batch)
BatchNorm:对特征归一化(垂直featrue)
在时序模型中，每个样本的长度(n)可能发生变化,导致BN计算均值方差时抖动较大
LN在样本内部计算，更加稳定

```text
      +--------+
    feature(d)/|
    +--------+ |
    |        | |
   batch     | +
    |        |/
    +-seq(n)-+
```

$d_{model}$=512

decoder:生成长为$m$的序列(只能一个一个生成)$[y_1,y_2,y_3...y_m]$，其输入为编码器在$t$时刻前的输出 自回归

scaled dot-product attention
query($n*d_k$),key($m*d_k$)和value($m*d_v$)
$QK^T$($n*m$)除以$\sqrt{d_k}$后，做softmax（对每一行），再乘$V$,得到的形状为$n*d_v$
mask 把$q_t$和$k_t$之后的值换成绝对值很大的负数，经过softmax后变成0
多头
投影
