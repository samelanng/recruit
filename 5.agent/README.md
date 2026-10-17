# Agent



1. 平台与模型

| 项目 | 值 |
| --- | --- |
| 平台 | 阿里云百炼（DashScope） |
| 接口协议 | OpenAI 兼容接口（Chat Completions） |
| Base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 模型 | `qwen-plus` |
| temperature | 任务一 0.3；任务二/三 0.1（信息抽取需要稳定输出） |

2. 环境配置

- Python 3.10 或以上
- 依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 内容：

```
openai>=1.0.0
python-dotenv>=1.0.0
```

3. API Key 配置

复制 `.env.example` 为 `.env`，填入自己在百炼控制台创建的 Key：

```
DASHSCOPE_API_KEY=sk-你的真实Key
```

`.env` 已在 `.gitignore` 中，不会提交到仓库。

4. 文件说明

| 文件 | 作用 |
| --- | --- |
| `task1.py` | 任务一：完成一次 API 调用并打印回答 |
| `task2.py` | 任务二：校园信息助手（输入文本 → 三字段固定格式输出） |
| `task3.py` | 任务三：同一输入下，"简单提示词"与"结构化提示词"的对比 |
| `results.md` | 任务二的三类测试输入与真实输出记录 |
| `.env.example` | 环境变量模板（只有变量名） |
| `.gitignore` | 忽略 `.env`、`__pycache__` 等 |

5. 运行方法

```bash
# 任务一
python task1.py

# 任务二（运行后输入一段校园文本，回车提交）
python task2.py

# 任务三（对同一段讲座通知跑两组提示词并打印结果）
python task3.py
```

6. 提示词设计思路

任务二使用的提示词包含三个要素：

- 角色：告诉模型它是校园信息助手、服务对象是大学生；
- 输出格式：只给三个空字段（`一句话总结 / 关键信息 / 待办事项`），字段名和顺序固定；
- 约束条件：只输出这三行、不要解释；原文没有的信息写"未提及"，不允许编造；待办事项只列原文明确要求做的事。



