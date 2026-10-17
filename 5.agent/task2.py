import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

SYSTEM_PROMPT = """你是一个校园信息助手，服务对象是大学生。

请严格按下面三行输出，字段名和顺序不能变，不要输出任何解释、前言或结语：

一句话总结：
关键信息：
待办事项：

要求：
1. 一句话总结：用一句话概括这段文本在讲什么。
2. 关键信息：列出时间、地点、对象、要求等关键要素，用顿号分隔；原文没有写的内容写"未提及"。
3. 待办事项：只列出原文明确要求学生做的事，用 1. 2. 3. 编号；没有则写"无"。不要添加原文没有的要求。
4. 如果输入是学习问题而不是通知，也按同一格式回答：总结写问题核心，关键信息写涉及的概念，待办事项写需要掌握或理解的内容。"""


def analyze(text: str) -> str:
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    print("请输入一段校园文本，直接回车结束：")
    print("-" * 50)
    print(analyze(input()))
