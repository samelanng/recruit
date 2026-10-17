import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL = "qwen-plus"

TEST_TEXT = """周六下午三点将在图书馆报告厅举办人工智能讲座，
主讲人是计算机学院张老师，现场可签到加第二课堂学分，请提前十分钟入场。"""

#简单提示词：没有角色、格式和约束
SIMPLE = "请总结以下通知：\n" + TEST_TEXT

#包含角色+输出格式+约束条件
STRUCTURED = (
    "你是一个面向大学生的校园信息助手。\n"
    "请从下面的文本中提取信息，并严格按以下格式输出：\n"
    "一句话总结：<一句话概括>\n"
    '关键信息：<时间、地点、对象、要求，用顿号分隔；没有的写"未提及">\n'
    '待办事项：<1. 2. 3. 编号列出；没有则写"无">\n'
    "约束条件：\n"
    "1. 不要输出格式之外的内容；\n"
    '2. 原文没有的信息写"未提及"，禁止编造；\n'
    "3. 用词客观，不要添加评价与建议。\n"
    "文本：\n" + TEST_TEXT
)

def run(prompt: str, use_system: str | None = None) -> str:
    messages = []
    if use_system:
        messages.append({"role": "system", "content": use_system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def compare(a_prompt: str, b_prompt: str, use_system: str | None = None):
    a = run(a_prompt, use_system)
    b = run(b_prompt, use_system)

    print("[简单提示词] 字数：%d" % len(a))
    print(a)
    print()

    print("[结构化提示词] 字数：%d" % len(b))
    print(b)
    return a, b

def run(prompt: str, use_system: str | None = None) -> str:
    messages = []
    if use_system:
        messages.append({"role": "system", "content": use_system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,  
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":    
    compare(SIMPLE, STRUCTURED)