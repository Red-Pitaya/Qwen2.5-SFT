import hashlib
import json
import os
import random
import time
from collections import Counter

import numpy as np

from datasets import DatasetDict, load_dataset

REVISION = "e9c87b4"
SUBSET_ROWS = 1_000_000

print(f"  - revision: {REVISION}")
print(f"  - 子集大小: {SUBSET_ROWS:,} 条")

start_time = time.time()

train_ds = load_dataset(
    "Open-Orca/OpenOrca",
    split=f"train[:{SUBSET_ROWS}]",
    revision=REVISION,
)
dataset = DatasetDict({"train": train_ds})

load_time = time.time() - start_time
print(f"数据加载完成！load_time: {load_time:.1f} 秒")
print(f"dataset: {dataset}")
print(f"column_names: {dataset['train'].column_names}")
print(f"len: {len(dataset['train']):,} 条")

print("\n--- sample(3)---")
for i in range(min(3, len(dataset["train"]))):
    sample = dataset["train"][i]
    print(f"\nsample {i + 1}:")
    print(f"  id: {sample.get('id', 'N/A')}")
    print(f"  system_prompt: {str(sample.get('system_prompt', ''))[:100]}...")
    print(f"  question: {str(sample.get('question', ''))[:100]}...")
    print(f"  response: {str(sample.get('response', ''))[:100]}...")


data = dataset["train"]

question_lengths = [len(str(sample["question"])) for sample in data]
response_lengths = [len(str(sample["response"])) for sample in data]

null_question = sum(
    1
    for sample in data
    if not sample.get("question") or str(sample["question"]).strip() == ""
)
null_response = sum(
    1
    for sample in data
    if not sample.get("response") or str(sample["response"]).strip() == ""
)
null_system = sum(
    1
    for sample in data
    if not sample.get("system_prompt")
    or str(sample.get("system_prompt", "")).strip() == ""
)

print(f"  question NULL: {null_question:,}")
print(f"  response NULL: {null_response:,}")
print(f"  system_prompt NULL: {null_system:,}")

print("\nquestion 长度分布（字符数）")
print(f"  最小值: {min(question_lengths)}")
print(f"  最大值: {max(question_lengths)}")
print(f"  平均值: {sum(question_lengths) / len(question_lengths):.1f}")
print(f"  中位数: {sorted(question_lengths)[len(question_lengths) // 2]}")

print("\nresponse 长度分布（字符数）")
print(f"  最小值: {min(response_lengths)}")
print(f"  最大值: {max(response_lengths)}")
print(f"  平均值: {sum(response_lengths) / len(response_lengths):.1f}")
print(f"  中位数: {sorted(response_lengths)[len(response_lengths) // 2]}")

system_prompts = [str(sample.get("system_prompt", ""))[:200] for sample in data]
system_counter = Counter(system_prompts)
print("\nsystem_prompt 类型分布（前10种）")
for prompt, count in system_counter.most_common(10):
    print(f"  [{count:>8,}条] {prompt[:80]}...")


cleaned_data = [dict(sample) for sample in data]
original_count = len(cleaned_data)
print(f"Before clean: {original_count:,} 条")

print("\n 格式清洗（删除空值/缺失字段）")
before_count = len(cleaned_data)
cleaned_data = [
    sample
    for sample in cleaned_data
    if sample.get("question")
    and str(sample["question"]).strip() != ""
    and sample.get("response")
    and str(sample["response"]).strip() != ""
]
after_count = len(cleaned_data)
print(f"  删除 {before_count - after_count:,} 条空值数据，剩余: {after_count:,} 条")

print("\n 长度过滤")
before_count = len(cleaned_data)
MIN_QUESTION_LEN = 10
MIN_RESPONSE_LEN = 20
MAX_TOTAL_LEN = 4000
cleaned_data = [
    sample
    for sample in cleaned_data
    if len(str(sample["question"]).strip()) >= MIN_QUESTION_LEN
    and len(str(sample["response"]).strip()) >= MIN_RESPONSE_LEN
    and (len(str(sample["question"])) + len(str(sample["response"]))) <= MAX_TOTAL_LEN
]
after_count = len(cleaned_data)
print(f"  删除 {before_count - after_count:,} 条，剩余: {after_count:,} 条")

print("\n质量过滤")
before_count = len(cleaned_data)
LOW_QUALITY_PATTERNS = [
    "i cannot",
    "i can't",
    "i'm unable to",
    "i am unable to",
    "as an ai",
    "as a language model",
    "as an artificial intelligence",
    "i don't have the ability",
    "i do not have the ability",
    "i'm not able to",
    "i apologize, but i cannot",
    "sorry, but i can't",
    "i'm sorry, but as an",
    "this is not something i can",
]


def is_low_quality(response_text):
    # 只检查前200字符
    response_lower = response_text.lower().strip()[:200]
    for pattern in LOW_QUALITY_PATTERNS:
        if pattern in response_lower:
            return True
    return False


cleaned_data = [
    sample for sample in cleaned_data if not is_low_quality(str(sample["response"]))
]
after_count = len(cleaned_data)
print(f"  删除 {before_count - after_count:,} 条低质量数据，剩余: {after_count:,} 条")

print("\n去重")
before_count = len(cleaned_data)
seen_hashes = set()
deduplicated_data = []
for sample in cleaned_data:
    question_text = str(sample["question"]).strip().lower()
    question_hash = hashlib.md5(question_text.encode("utf-8")).hexdigest()
    if question_hash not in seen_hashes:
        seen_hashes.add(question_hash)
        deduplicated_data.append(sample)
cleaned_data = deduplicated_data
after_count = len(cleaned_data)
print(f"  删除 {before_count - after_count:,} 条重复数据，剩余: {after_count:,} 条")

print("\n一致性检查")
for sample in cleaned_data:
    sample["question"] = str(sample["question"]).strip()
    sample["response"] = str(sample["response"]).strip()
    if (
        not sample.get("system_prompt")
        or str(sample.get("system_prompt", "")).strip() == ""
    ):
        sample["system_prompt"] = ""
    else:
        sample["system_prompt"] = str(sample["system_prompt"]).strip()
print(f"  格式标准化完成，当前数据量: {len(cleaned_data):,} 条")

print(f"\n基础清洗完成: {original_count:,} → {len(cleaned_data):,} 条")


TARGET_TOTAL = 200_000  # 目标采样总量（任务要求 20-35 万范围）
K_CLUSTERS = 500  # 聚类数量（清洗后 ~30 万数据，每簇平均 ~600 条）
RANDOM_SEED = 42  # 固定随机种子
MIN_PER_CLUSTER = 20  # 每簇最少采样数（保证极小簇也有代表）

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("\nSentence Embedding 编码")
embed_start = time.time()

from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 提取所有 question 文本用于编码
# 截取前 256 字符以控制编码时间（all-MiniLM-L6-v2 最大处理 256 tokens）
sentences = [sample["question"][:256] for sample in cleaned_data]

print(f"正在编码 {len(sentences):,} 条文本（batch_size=256）...")
embeddings = embed_model.encode(
    sentences,
    show_progress_bar=True,
    batch_size=256,  # 大 batch 加速 GPU 推理
    normalize_embeddings=True,  # L2 归一化，便于后续距离计算
)
embeddings = np.array(embeddings)

embed_time = time.time() - embed_start
print(f"编码完成！维度: {embeddings.shape}，耗时: {embed_time:.1f} 秒")

print("\nMiniBatchKMeans 聚类")
cluster_start = time.time()

from sklearn.cluster import MiniBatchKMeans

print(f"聚类参数: K={K_CLUSTERS}, batch_size=2048")
kmeans = MiniBatchKMeans(
    n_clusters=K_CLUSTERS,
    batch_size=2048,
    random_state=RANDOM_SEED,
    n_init=3,  # 运行 3 次取最优，平衡质量与速度
    max_iter=100,  # 最大迭代次数
)
kmeans.fit(embeddings)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

cluster_time = time.time() - cluster_start
print(f"聚类完成！耗时: {cluster_time:.1f} 秒")

cluster_sizes = np.bincount(labels, minlength=K_CLUSTERS)
print(f"  簇数量: {K_CLUSTERS}")
print(f"  最大簇: {cluster_sizes.max():,} 条")
print(f"  最小簇: {cluster_sizes.min():,} 条")
print(f"  平均簇: {cluster_sizes.mean():.0f} 条")
print(f"  中位数: {np.median(cluster_sizes):.0f} 条")
print(f"  空簇数: {np.sum(cluster_sizes == 0)}")

print("\n最大的 10 个簇")
largest_clusters = np.argsort(cluster_sizes)[::-1][:10]
for rank, cid in enumerate(largest_clusters):
    cluster_mask = labels == cid
    cluster_indices = np.where(cluster_mask)[0]
    cluster_embeddings = embeddings[cluster_indices]
    dists = np.linalg.norm(cluster_embeddings - centers[cid], axis=1)
    closest_idx = cluster_indices[np.argsort(dists)[:3]]

    print(f"\n  排名 {rank + 1}: 簇 {cid}（{cluster_sizes[cid]:,} 条）")
    for idx in closest_idx:
        print(f"    代表样本: {cleaned_data[idx]['question'][:80]}...")

print("\n对数平衡采样 ")
print(f"目标采样总量: {TARGET_TOTAL:,} 条")

# 计算各簇的对数权重
# 对空簇或极小簇做保护（至少 1 条）
log_weights = np.log(np.maximum(cluster_sizes, 1) + 1)  # +1 避免 log(1)=0
total_log_weight = log_weights.sum()

# 按对数权重分配配额
quotas = np.maximum(
    (log_weights / total_log_weight * TARGET_TOTAL).astype(int),
    MIN_PER_CLUSTER,  # 每簇最少 20 条
)

# 确保配额不超过簇本身大小
quotas = np.minimum(quotas, cluster_sizes)

# 微调总量使其接近目标（按比例缩放）
current_total = quotas.sum()
if current_total > TARGET_TOTAL:
    # 按比例缩减
    scale = TARGET_TOTAL / current_total
    quotas = np.maximum(
        (quotas * scale).astype(int), np.minimum(MIN_PER_CLUSTER, cluster_sizes)
    )
elif current_total < TARGET_TOTAL:
    # 按比例扩大，但不超过簇大小
    deficit = TARGET_TOTAL - current_total
    expandable = cluster_sizes - quotas
    expand_weights = (
        expandable / expandable.sum()
        if expandable.sum() > 0
        else np.zeros_like(expandable)
    )
    extra = (expand_weights * deficit).astype(int)
    quotas = np.minimum(quotas + extra, cluster_sizes)

actual_total = quotas.sum()
print(f"实际采样总量: {actual_total:,} 条（目标 {TARGET_TOTAL:,}）")
print(f"非空簇数: {np.sum(quotas > 0)}/{K_CLUSTERS}")
print(f"配额最大: {quotas.max():,} 条")
print(f"配额最小（非零）: {quotas[quotas > 0].min():,} 条")
print(f"配额中位: {np.median(quotas[quotas > 0]):.0f} 条")

print("\n簇内中心点采样")
sample_start = time.time()

final_indices = []

for cid in range(K_CLUSTERS):
    quota = quotas[cid]
    if quota == 0:
        continue

    # 获取该簇的所有样本索引
    cluster_indices = np.where(labels == cid)[0]

    if len(cluster_indices) <= quota:
        # 簇大小不足配额，全部选取
        final_indices.extend(cluster_indices.tolist())
    else:
        # 计算各样本到簇中心的距离
        cluster_embeddings = embeddings[cluster_indices]
        dists = np.linalg.norm(cluster_embeddings - centers[cid], axis=1)

        # 按距离升序排列（离中心近的优先）
        sorted_local = np.argsort(dists)
        selected = cluster_indices[sorted_local[:quota]]
        final_indices.extend(selected.tolist())

sample_time = time.time() - sample_start
print(f"采样完成！耗时: {sample_time:.1f} 秒")
print(f"最终采样数量: {len(final_indices):,} 条")

final_data = [cleaned_data[i] for i in final_indices]

random.shuffle(final_data)

cleaned_data = final_data

final_count = len(cleaned_data)

print(f"\n原始数据量: {original_count:>10,} 条")
print(f"基础清洗后:  {len(deduplicated_data):>10,} 条")
print(f"聚类采样后（最终）: {final_count:>10,} 条")
print(f"总保留比例: {final_count / original_count * 100:>9.1f}%")

final_q_lens = [len(sample["question"]) for sample in cleaned_data]
final_r_lens = [len(sample["response"]) for sample in cleaned_data]

print("\nquestion 长度分布")
print(f"最小值: {min(final_q_lens)}")
print(f"最大值: {max(final_q_lens)}")
print(f"平均值: {sum(final_q_lens) / len(final_q_lens):.1f}")

print("\nresponse 长度分布")
print(f"最小值: {min(final_r_lens)}")
print(f"最大值: {max(final_r_lens)}")
print(f"平均值: {sum(final_r_lens) / len(final_r_lens):.1f}")

# 聚类多样性分析：对比最终数据中各簇的覆盖情况
final_labels = labels[final_indices]
final_cluster_counts = np.bincount(final_labels, minlength=K_CLUSTERS)
covered_clusters = np.sum(final_cluster_counts > 0)
print(
    f"覆盖簇数: {covered_clusters}/{K_CLUSTERS} ({covered_clusters / K_CLUSTERS * 100:.1f}%)"
)
print(f"最大簇采样: {final_cluster_counts.max():,} 条")
print(
    f"最小簇采样（非零）: {final_cluster_counts[final_cluster_counts > 0].min():,} 条"
)
print(
    f"Gini 不均衡度: {1 - (final_cluster_counts[final_cluster_counts > 0] / final_cluster_counts.sum()).max():.4f}"
)

os.makedirs("data", exist_ok=True)

output_path = "data/openorca_cleaned_raw.json"
print(f"保存到 {output_path}")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=None)

file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"保存完成。文件{file_size_mb:.1f} MB")
print(f"数据{final_count:,} 条")

cluster_stats = {
    "version": "v2_cluster",
    "k_clusters": K_CLUSTERS,
    "target_total": TARGET_TOTAL,
    "actual_total": int(actual_total),
    "original_count": original_count,
    "after_basic_clean": len(deduplicated_data),
    "cluster_sizes": cluster_sizes.tolist(),
    "quotas": quotas.tolist(),
    "covered_clusters": int(covered_clusters),
    "embed_model": "all-MiniLM-L6-v2",
    "embed_time_sec": round(embed_time, 1),
    "cluster_time_sec": round(cluster_time, 1),
}

stats_path = "data/cluster_stats.json"
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(cluster_stats, f, ensure_ascii=False, indent=2)
print(f"聚类统计信息保存到{stats_path}")

total_time = time.time() - start_time
print(f"\ntotal time: {total_time / 60:.1f} 分钟")