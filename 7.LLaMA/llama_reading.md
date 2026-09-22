# LLaMA Reading

- 1.LLaMA 和原始 Transformer 的关系是什么？​
LLaMA 使用的是哪一类 Transformer 架构？它的训练目标是什么？为什么“预测下一个 token”可以成为生成文本、回答问题等能力的基础？​
- 2.LLaMA 相比你已经学习的 Transformer 做了哪些关键改动？​
请说明 RMSNorm、SwiGLU、RoPE 分别解决或改善了什么问题。无需推导公式，但需要说明它们在模型中的作用。​
- 3.为什么 LLaMA 不只是“参数更多的 Transformer”？​
论文如何处理训练数据来源、数据配比、训练 token 数与模型参数量之间的关系？为什么足够多且高质量的数据对小一些的模型也很重要？​
- 4.论文中的 zero-shot / few-shot 评测是怎样完成的？​
选择论文中的一个评测任务说明：模型如何把一个分类、问答或推理任务转化为文本生成任务？这个过程是否更新了模型参数？它和微调有什么本质区别？​
- 5.预训练模型为什么不能直接等同于聊天助手？​
结合论文中的 instruction fine-tuning 内容，说明 Base Model 与 Instruct Model 的区别。指令微调改变了模型的哪些部分，又没有改变哪些部分？
