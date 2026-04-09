import pandas as pd
import numpy as np


def main():
    # 1. 读取 Ragas 生成的成绩单
    csv_file = "result/experiment_results_log_v2.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"找不到文件 {csv_file}，请确认文件名和路径。")
        return

    # 2. 检查必要的列是否存在
    required_cols = ['context_precision', 'context_recall', 'answer_correctness']
    for col in required_cols:
        if col not in df.columns:
            print(f"错误：成绩单中缺少必需的列 '{col}'")
            return

    # 3. 定义 F1 计算逻辑 (防止除以 0 的错误)
    def compute_f1(row):
        p = row['context_precision']
        r = row['context_recall']
        # 处理空值或两者都为 0 的情况
        if pd.isna(p) or pd.isna(r) or (p + r) == 0:
            return 0.0
        # 标准 F1 调和平均公式
        return 2 * (p * r) / (p + r)

    # 4. 计算每道题的 F1 值，并作为新列加入表格
    df['retrieval_f1'] = df.apply(compute_f1, axis=1)

    # 可以选择把带有 F1 值的新表格保存下来
    df.to_csv("experiment_results_with_f1.csv", index=False)

    # 5. 计算并打印大盘平均分
    print("\n" + "=" * 40)
    print("🏆 基线系统 (Naive RAG) 最终评测大盘")
    print("=" * 40)

    mean_precision = df['context_precision'].mean()
    mean_recall = df['context_recall'].mean()
    mean_f1 = df['retrieval_f1'].mean()
    mean_correctness = df['answer_correctness'].mean()

    print(f"1. 上下文精确率 (Context Precision): {mean_precision:.4f}")
    print(f"2. 上下文召回率 (Context Recall):    {mean_recall:.4f}")
    print(f"3. 检索综合指标 (Retrieval F1):    {mean_f1:.4f}  <-- 简历核心指标")
    print("-" * 40)
    print(f"4. 端到端生成准确度 (Answer Correctness): {mean_correctness:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()