# config.py
# 负责加载环境变量，并提供 LLM、FastLLM、Embedding 模型的统一初始化入口。
# 所有模块通过调用此处的工厂函数获取模型实例，避免重复配置。

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

def get_fast_llm():
    return ChatOpenAI(
        model="glm-4-flash",
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_URL"),
        temperature=1.0,
        max_tokens=65536,
    )

def get_embeddings():
    return OpenAIEmbeddings(
        model="Embedding-3",
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_URL"),
        check_embedding_ctx_length=False,
    )
