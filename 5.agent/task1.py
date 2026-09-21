import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "你是一个乐于助人的助手，回答尽量简洁。"},
        {"role": "user", "content": "用一句话解释什么是机器学习。"},
    ],
    temperature=0.3,
)

print(resp.choices[0].message.content)
