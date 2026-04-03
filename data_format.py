import json
import os
import time

input_path = "data/openorca_cleaned_raw.json"

if not os.path.exists(input_path):
    print(f"找不到文件 {input_path}")
    exit(1)

start_time = time.time()

with open(input_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

load_time = time.time() - start_time
print(f"加载完成。耗时: {load_time:.1f} 秒")
print(f"数据{len(raw_data):,} 条")

alpaca_data = []

for i, sample in enumerate(raw_data):
    alpaca_sample = {
        "instruction": sample['question'],     # 用户指令/问题
        "input": "",                            # 额外输入（OpenOrca 没有独立的 input 字段，设为空）
        "output": sample['response'],           # 期望的模型回复
    }

    # 如果有 system_prompt 且不为空，添加 system 字段
    # LlamaFactory 支持可选的 system 字段来设置系统提示
    if sample.get('system_prompt') and sample['system_prompt'].strip():
        alpaca_sample["system"] = sample['system_prompt']

    alpaca_data.append(alpaca_sample)

    if (i + 1) % 100000 == 0:
        print(f"  已转换: {i + 1:,} / {len(raw_data):,} 条")

print(f"共 {len(alpaca_data):,} 条数据")

print("\n转换后sample（前3条）---")
for i in range(min(3, len(alpaca_data))):
    print(f"\nsample {i+1}:")
    print(f"  system: {str(alpaca_data[i].get('system', ''))[:80]}...")
    print(f"  instruction: {alpaca_data[i]['instruction'][:80]}...")
    print(f"  input: '{alpaca_data[i]['input']}'")
    print(f"  output: {alpaca_data[i]['output'][:80]}...")

with_system = sum(1 for d in alpaca_data if 'system' in d)
print(f"\n包含 system 字段的数据: {with_system:,} 条 ({with_system/len(alpaca_data)*100:.1f}%)")

output_path = "data/openorca_cleaned.json"
print(f"保存最终格式数据到 {output_path}")

start_time = time.time()
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=None)

save_time = time.time() - start_time
file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"保存完成。耗时: {save_time:.1f} 秒，文件大小: {file_size_mb:.1f} MB")

dataset_info = {
    "openorca_cleaned": {
        "file_name": "openorca_cleaned.json",  # 数据文件路径
        "columns": {
            "prompt": "instruction",   # 用户指令字段名
            "query": "input",          # 额外输入字段名
            "response": "output",      # 模型回复字段名
            "system": "system"         # 系统提示字段名
        }
    }
}

dataset_info_path = "data/dataset_info.json"
with open(dataset_info_path, 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, ensure_ascii=False, indent=2)

print(f"数据集注册文件保存到{dataset_info_path}")
print("注册数据集名称为 openorca_cleaned")
print("\nLlamaFactory 训练时使用以下配置引用此数据集:")
print('  dataset: openorca_cleaned')
print('  dataset_dir: data')

with open(output_path, 'r', encoding='utf-8') as f:
    verify_data = json.load(f)

print(f"验证读取{len(verify_data):,} 条数据")

required_fields = ['instruction', 'output']
missing_count = 0
for sample in verify_data[:1000]:  # 只检查前1000条
    for field in required_fields:
        if field not in sample or not sample[field]:
            missing_count += 1
            break

if missing_count == 0:
    print("字段验证通过")
else:
    print(f"字段验证失败，{missing_count} 条样本缺少必要字段")

with open(dataset_info_path, 'r', encoding='utf-8') as f:
    verify_info = json.load(f)

if 'openorca_cleaned' in verify_info:
    print("openorca_cleaned 已在 dataset_info.json 中注册")
else:
    print("注册验证失败")

print(f"训练数据: {output_path} ({len(verify_data):,} 条)")
print(f"注册文件: {dataset_info_path}")
