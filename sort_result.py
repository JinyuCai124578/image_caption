import json


# 读取 result.json 文件
with open('result.json', 'r') as f:
    results = json.load(f)

# 定义指标
score_key = 'CIDEr'  # 可以选择 BLEU, ROUGE, METEOR, CIDEr

# 提取并排序结果
sorted_results = sorted(
    results,
    key=lambda x: sum(x['scores']['Bleu'])+x['scores']['Rouge']+x['scores']['METEOR']+x['scores']['CIDEr'],
    reverse=True  # 从高到低排序
)

# 获取最好和最坏的五个样例
best_samples = sorted_results[:5]
worst_samples = sorted_results[-5:]

# 打印结果
print("Best Samples:")
for sample in best_samples:
    print(json.dumps(sample, indent=4))

print("\nWorst Samples:")
for sample in worst_samples:
    print(json.dumps(sample, indent=4))

#保存到sorted_result.json
with open('sorted_result.json', 'w') as f:
    json.dump(sorted_results, f, indent=4)