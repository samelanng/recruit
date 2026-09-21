from pathlib import Path
import numpy as np
import argparse #解析命令行参数
import os
import re
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
KB_DIR=Path(__file__).parent/"knowledge_base"#__file__之当前文件
CHUNK_SIZE=300#文本块最大目标长度
CHUNK_OVERLAP=60#相邻文本块之间的重叠长度，防止重要信息刚好被切断
DEFAULT_TOPK=4#默认检索数量（相关的文本块）
EMBED_MODEL="text-embedding-v3"
LOCAL_EMBED_MODEL="BAAI/bge-small-zh-v1.5"
CHAT_MODEL="qwen-plus"
client=OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
_local_model=None

def get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model=SentenceTransformer(LOCAL_EMBED_MODEL)
    return _local_model
    
def load_documents(kb_dir:Path):
    docs=[]
    for path in sorted(kb_dir.glob("*.md")):
        docs.append((path.name,path.read_text(encoding="utf-8")))
    return docs

def split_documents(docs,chunk_size=300,overlap=60):
    chunks=[]
    for name,text in docs:
        text=text.replace("\r\n","\n").strip()#换行符的替换，strip()删除字符串开头结尾的空白字符
        paras=[p.strip() for p in re.split(r"\n\s*\n",text) if p.strip()]#按段落切分
        cur=""
        for p in paras:
            if len(p)>chunk_size:
                if cur:#有遗留内容
                    chunks.append((name,cur))
                    cur=""
                for i in range(0,len(p),chunk_size-overlap):
                    chunks.append((name,p[i:i+chunk_size]))
                    if i+chunk_size>=len(p):
                        break
                continue
            if not cur:
                cur=p
            elif len(cur)+len(p)+2<=chunk_size:
                cur=cur+"\n\n"+p
            else:
                chunks.append((name,cur))
                cur=(cur[-overlap:]+"\n\n"+p) if overlap else p
        if cur:
            chunks.append((name,cur))
    return chunks
#将文本转换为向量，形成向量索引
def build_index(chunks,backend="auto"):
    texts=[c[1] for c in chunks]#c[0]为文本名，c[1]为文本内容
    if backend in("auto","local"):
        try:
            model=get_local_model()
            vecs=model.encode(texts,normalize_embeddings=True)#转换成向量并归一化(将每个向量的长度调整为1，方便后面通过点积计算余弦相似度)
            return np.asarray(vecs,dtype=np.float32),"local:%s"% LOCAL_EMBED_MODEL
        except Exception:
            if backend=="local":
                raise
            print("本地Embedding不可用,改用API Embedding")
    vectors=[]
    #批量处理
    for i in range(0,len(texts),10):
        batch=texts[i:i+10]
        resp=client.embeddings.create(model=EMBED_MODEL,input=batch)
        vectors.extend([d.embedding for d in resp.data])
    arr=np.asarray(vectors,dtype=np.float32)
    arr/=np.linalg.norm(arr,axis=1,keepdims=True)#归一化(np.linalg.norm计算向量长度，axis=1按行计算)
    return arr, "api:%s" % EMBED_MODEL

def embed_query(question, backend):
    if backend.startswith("local"):
        v=get_local_model().encode([question], normalize_embeddings=True)#问题向量化
        return np.asarray(v, dtype=np.float32)
    resp=client.embeddings.create(model=EMBED_MODEL, input=[question])
    v=np.asarray([resp.data[0].embedding], dtype=np.float32)
    v/=np.linalg.norm(v, axis=1, keepdims=True)
    return v


def retrieve(question,chunks,index,backend,topk=4):
    qv=embed_query(question,backend)
    scores=(index@qv.T).ravel()#矩阵乘法计算向量相似度，raval()把二维数组转化成一维
    order=np.argsort(-scores)[:topk]#按照相似度从高到低排列，并取出前topk个文本块的下标
    return [{"score":float(scores[i]),"source":chunks[i][0],"text":chunks[i][1]} for i in order]

SYSTEM_PROMPT = (
    "你是一个严谨的资料问答助手。你只能依据用户提供的资料回答问题，"
    "禁止使用任何资料以外的知识，包括常识和你自己的经验。"
    "资料中没有提到的信息，一律视为不存在。"
)

PROMPT = """资料：
{context}

问题：{question}

回答要求：
1. 只依据上面的资料作答，不要补充资料中没有的内容，也不要给出资料中没有的建议或推荐。
2. 如果资料无法回答该问题，只输出"资料中未提及"，不要解释原因。
3. 回答控制在 150 字以内，用简洁的句子，不要使用 emoji、表格或加粗。
4. 最后按下面的格式列出你实际用到的资料文件名，只写纯文件名：
参考资料：
- 文件名
"""

def generate_answer(question, hits):
    context="\n\n".join("【%s】\n%s" % (h["source"], h["text"]) for h in hits)#将多个检索结果拼接成一段完整的
    resp=client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":PROMPT.format(question=question, context=context)},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()#第一个候选回答

def main():
    #添加参数
    ap=argparse.ArgumentParser()
    ap.add_argument("--question",type=str,default=None)
    ap.add_argument("--topk",type=int,default=DEFAULT_TOPK)
    ap.add_argument("--backend",default="auto",choices=["auto","api","local"])
    ap.add_argument("--chunk-size",type=int,default=CHUNK_SIZE)
    ap.add_argument("--overlap",type=int,default=CHUNK_OVERLAP)
    args=ap.parse_args()
    docs=load_documents(KB_DIR)
    chunks=split_documents(docs,args.chunk_size,args.overlap)
    index,backend=build_index(chunks,args.backend)
    print("已读取文件：%s"%"、".join("%s(%d字)"%(n,len(t)) for n,t in docs))
    print("共切分为%d个文本块,Embedding后端:%s"%(len(chunks),backend))
    question=args.question
    one_shot=args.question is not None
    while True:
        if question is None:
            try:
                question=input("\n请输入问题（回车退出）：").strip()
            except(EOFError,KeyboardInterrupt):
                break
        if not question:
            break
        hits=retrieve(question,chunks,index,backend,args.topk)
        print("\n检索结果")
        for i,h in enumerate(hits,1):
            preview=h["text"].replace("\n"," ")[:80]
            print("Top %d 来源：%s（相似度%.4f）\n  文本：%s..."%(i,h["source"],h["score"],preview))
        print("\n最终回答")
        print(generate_answer(question,hits))
        if one_shot:
            break
        question=None

if __name__=="__main__":
    main()
