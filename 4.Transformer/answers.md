# Transformer

## 理论问答

- 1.在 Transformer 出现之前，RNN、LSTM 等序列模型主要存在哪些局限？

①RNN等序列模型是时序神经网络，第t步必须等第t−1步算完，难以并行计算，时间消耗长，计算性能差；②容易丢失早期的信息，若要通过增大$h_t$来降低丢失早期信息的可能性，又会增大内存消耗；③ 固定大小的隐状态是信息瓶颈，整段序列被压成一个向量；④任意两个位置之间要经过O(n)步才能交互，路径太长；⑤长距离依赖靠隐状态逐步传递，梯度容易消失或爆炸，LSTM 的门控只是缓解。

- 2.Transformer 与 RNN 在处理序列时，最大的结构区别是什么？
Transform不使用循环层和卷积，只依赖注意力机制，用自注意力一次性让序列中所有位置两两直接交互，完全可并行，代价是需要额外的 Positional Encoding 来注入顺序信息。
- 3.Transformer 中最核心的公式之一是：​
  
$$
\text{Attention}(Q,K,V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}}\right)V
$$
​a.$Q$、$K$、$V$分别起到什么作用？
$Q$:查询，表示当前位置想找什么信息； $K$：答案，表示每个位置能提供什么信息 $T$：值，真正被加权聚合的内容。
b.$QK^T$得到的结果是什么？
相似度 权重
得到一个seq_len × seq_len的分数矩阵，第$i$行第$j$列是"第$j$个位置对第$j$个位置的关注程度"（内积越大表示越相关），是未归一化的相似度。
c.softmax在这里做了什么？
沿最后一维把分数变成非负、且每行加起来等于1的权重分布，这样输出就是V的加权平均，权重可解释为注意力分配比。
d.为什么要除以$\sqrt{d_k}$？如果不缩放，可能带来什么问题？
点积运算结果是$d_k$个乘积项之和，若各分量独立且方差为1，其结果的方差正比于$d_k$，若$d_k$比较大的时候，点积运算的结果也会较大，各结果之间的差距也会增大，通过softmax后大的值会更加靠近1，另外的值会更加靠近0，向两端靠拢，导致计算梯度时梯度较小。

- 4.为什么要使用多个 Attention Head？
多输出通道，识别不同的模式，增加学习参数
- 5.为什么 Transformer 需要 Positional Encoding？为什么作者使用$sin$和$cos$函数来表示位置信息？
在输入中加入时序信息​。自注意力对输入顺序是置换等变的，把序列打乱，输出只是跟着打乱，模型本身感知不到顺序，所以必须显式注入位置信息。
为什么用 sin/cos：① 不需要额外学习参数，值域有界，长短序列都稳定；② 不同频率的正弦波组合能唯一编码每个位置，且PE(pos+k)可以表示成PE(pos)的线性变换（旋转关系），模型更容易学到相对距离；③ 因为函数形式固定，可以外推到训练时没见过的更长序列。
- 6.为什么 Decoder 中需要使用 Masked Self-Attention？​
避免看到$t$时刻以后的输入，保证训练和预测的一致，用上三角掩码把未来位置的分数置为 −∞，softmax 后权重为0，保证第$t$个位置只能看到前$t$个位置。
- 7.Decoder 和 Encoder 相比，多了什么 Attention 结构？​
多了 Cross-Attention（Encoder-Decoder Attention），其中Q来自解码器自身，K、V来自编码器的输出，这是解码器读取源句子的唯一通道。此外解码器的 Self-Attention 是带掩码的，而编码器的是全可见的。
- 8.Residual Connection 和 Layer Normalization 在 Transformer 中分别起什么作用？
残差连接提供一条恒等通路，让梯度可以直接回传，缓解深层网络的梯度消失和退化问题，也让每层只需学习增量；Layer Normalization 对每个样本的特征维做归一化，稳定各层输入分布、加速收敛。相比 BatchNorm，它不依赖 batch 统计量，因此对变长序列、小 batch 更友好；这两者组合就是论文里的LayerNorm(x + Sublayer(x))。

learning_note
残差连接
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
