import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

SYSTEM_PROMPT = """你是一个校园信息助手，服务对象是大学生。
请严格按下面的格式输出，不要输出任何多余的解释：
一句话总结：<用一句话概括这段文本在讲什么>
关键信息：<时间、地点、对象、要求，用顿号分隔；原文没有的写"未提及">
待办事项：<1. 2. 3. 编号列出；没有则写"无">
硬性要求：字段名与顺序不能变；原文没有的时间地点一律写"未提及"，不许编造。"""

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
    print("请输入校园文本，直接回车结束：")
    print("-" * 50)
    print(analyze(input()))