import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, answer_correctness

# 引入你写的核心智能体
from app import IndustrialPDFLearningAgent
from langchain_core.messages import HumanMessage


def main():
    # 1. 初始化基线 Agent
    print("初始化 Naive RAG 基线模型...")
    agent = IndustrialPDFLearningAgent(user_id="eval_user")

    # [修改点]：批量加载评测集对应的所有论文
    pdf_files = [
        "S-Lora.pdf",
        "UniMoE.pdf",
        "MoLA.pdf",
        "MoRAL.pdf",
        "MoA.pdf",
        "Alpha_lora.pdf",
        "2309.05444v1.pdf",
        "2403.03432v1.pdf",
        "AdapterFusion.pdf",
        "Combining Modular Skills in Multitask Learning.pdf",
        "LoraHub.pdf"
    ]

    print("开始构建评测知识库...")
    for pdf in pdf_files:
        print(f"正在处理: {pdf}")
        result = agent.load_document(pdf)
        if not result["success"]:
            print(f"警告: {pdf} 加载失败 - {result['message']}")
    print("知识库构建完成！\n")

    # 2. 读取黄金评测集
    with open("result/eval_dataset.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    questions = []
    ground_truths = []
    answers = []
    contexts = []

    print(f"开始执行评测问答，共 {len(qa_pairs)} 道题...")
    for idx, pair in enumerate(qa_pairs):
        q = pair["question"]
        gt = pair["ground_truth"]

        # 隔离每次评测的上下文，避免历史记录污染
        config = {"configurable": {"thread_id": f"eval_thread_{idx}"}}
        inputs = {"query": q, "user_id": "eval_user", "messages": [HumanMessage(content=q)]}
        result = agent.app.invoke(inputs, config=config)

        ans = result["final_answer"]
        # Ragas 要求上下文是一个 List[str]
        retrieved_context = [result.get("retrieved_docs", "")]

        questions.append(q)
        ground_truths.append(gt)
        answers.append(ans)
        contexts.append(retrieved_context)
        print(f"完成 {idx + 1}/{len(qa_pairs)}")

    # 3. 构造 Ragas 需要的数据集格式
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # 4. 执行评估
    print("\n开始计算 Ragas 指标 (调用 LLM 作为裁判)...")
    eval_result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            answer_correctness
        ],
        raise_exceptions=False
    )

    # 5. 保存结果与实验参数记录
    df = eval_result.to_pandas()
    df["experiment_name"] = "Baseline_Naive_RAG"
    df["chunk_size"] = 800
    df["chunk_overlap"] = 150
    df["embedding_model"] = "text-embedding-3-small"

    # 追加保存到本地 CSV
    df.to_csv("experiment_results_log.csv", mode='a', header=not pd.io.common.file_exists("experiment_results_log.csv"),
              index=False)

    print("\n✅ 评测完成！整体基线分数：")
    print(eval_result)


if __name__ == "__main__":
    main()