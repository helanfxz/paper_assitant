import json
import os
from langchain_core.messages import HumanMessage
from app import IndustrialPDFLearningAgent


def main():
    print("=== [第一阶段：考生做题] ===")
    # 1. 初始化你的助手
    test_user = "eval_user"
    agent = IndustrialPDFLearningAgent(user_id=test_user)


    # 2. 批量自动化入库 (包含你最新加的 5 篇前沿论文)
    pdf_files = [
        "S-Lora.pdf", "UniMoE.pdf", "MoLA.pdf", "MoRAL.pdf", "MoA.pdf",
        "Alpha_lora.pdf", "Combining Modular Skills in Multitask Learning.pdf",
        "LoraHub.pdf", "AdapterFusion.pdf", "2309.05444v1.pdf", "2403.03432v1.pdf"
    ]
    """
    print("\n--- 开始批量构建知识库 ---")
    for pdf in pdf_files:
        result = agent.load_document(pdf)
        if result["success"]:
            print(f"✅ 成功: {result['message']}")
        else:
            print(f"❌ 失败: {pdf} - {result['message']}")
    """
    print("\n知识库构建完成！准备开始考试...\n")
    # 假设你已经把 PDF 入库了，这里可以直接跳过 load_document 步骤
    # 或者在这里按需加载新 PDF

    # 2. 读取黄金评测集
    dataset_path = "result/eval_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    results_to_save = []
    output_file = "result/generated_answers_cache_v2.json"

    # 如果有中断，支持断点续传（可选高级功能，这里提供基础落盘）
    print(f"开始做题，共 {len(qa_pairs)} 道题...")
    for idx, pair in enumerate(qa_pairs):
        q = pair["question"]
        gt = pair["ground_truth"]

        # 隔离每次评测的上下文
        config = {"configurable": {"thread_id": f"eval_thread_{idx}"}}
        inputs = {"query": q, "user_id": test_user, "messages": [HumanMessage(content=q)]}

        try:
            # 调用你的 PDF 助手生成答案
            result = agent.app.invoke(inputs, config=config)
            ans = result["final_answer"]
            retrieved_context = [result.get("retrieved_docs", "")]

            # 把这一题的结果记录下来
            results_to_save.append({
                "question": q,
                "ground_truth": gt,
                "answer": ans,
                "contexts": retrieved_context
            })
            print(f"✅ 完成 {idx + 1}/{len(qa_pairs)}")

            # 工业级习惯：每做完一题就落盘保存一次，防止中途崩溃
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results_to_save, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"❌ 第 {idx + 1} 题生成失败: {e}")

    print(f"\n🎉 做题完毕！所有答案已安全保存至 {output_file}")


if __name__ == "__main__":
    main()