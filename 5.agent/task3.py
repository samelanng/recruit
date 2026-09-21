import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

SYSTEM_PROMPT = "请总结以下通知：\n"

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