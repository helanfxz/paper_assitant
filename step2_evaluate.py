import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, answer_correctness
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from ragas.run_config import RunConfig
from dotenv import load_dotenv

load_dotenv()  # 确保环境变量里有 DASHSCOPE_API_KEY


def main():
    print("=== [第二阶段：裁判阅卷] ===")

    input_file = "result/generated_answers_cache_v2.json"

    # 1. 从硬盘读取做好的卷子
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            saved_results = json.load(f)
    except FileNotFoundError:
        print(f"找不到 {input_file}，请先运行 step1_generate.py！")
        return

    print(f"成功读取 {len(saved_results)} 份答案，准备开始评测...")

    # 2. 将数据转换为 Ragas 需要的格式
    data = {
        "question": [item["question"] for item in saved_results],
        "answer": [item["answer"] for item in saved_results],
        "contexts": [item["contexts"] for item in saved_results],
        "ground_truth": [item["ground_truth"] for item in saved_results]
    }
    dataset = Dataset.from_dict(data)

    # 3.
    print("正在配置裁判模型...")
    judge_embeddings = OpenAIEmbeddings(
            model="Embedding-3",
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url=os.getenv("ZHIPU_URL"),
            check_embedding_ctx_length=False
        )
    judge_llm = ChatOpenAI(
            model="glm-4-flash",
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url=os.getenv("ZHIPU_URL"),
            temperature=1.0,
            max_tokens=65536,
        )

    # ==========================================
    # 核心修复点：配置 Ragas 的运行参数
    # ==========================================
    print("配置限流控制 (防 429 报错)...")
    safe_config = RunConfig(
        max_workers=3,  # 【关键】把并发数降到 1，变成串行批改（慢但绝对安全）
        max_retries=10,  # 如果被限流，最多自动重试 10 次
        max_wait=60,  # 每次重试最大等待 60 秒
        timeout=120  # 给深度思考模型更长的超时时间
    )

    # 执行打分
    print("开始调用裁判模型进行逻辑打分（已开启安全限流模式，请耐心等待）...")
    eval_result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            answer_correctness
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=safe_config,  # 【关键】将安全配置传入 evaluate 函数
        raise_exceptions=False  # 哪怕有一道题彻底失败，也不要让整个程序崩溃
    )

    # 5. 保存成绩单
    df = eval_result.to_pandas()
    df["experiment_name"] = "New_RAG"

    output_csv = "experiment_results_log_v2.csv"
    df.to_csv(output_csv, mode='a', header=not pd.io.common.file_exists(output_csv), index=False)

    print(f"\n✅ 评测完成！成绩单已追加到 {output_csv}")
    print(eval_result)


if __name__ == "__main__":
    main()