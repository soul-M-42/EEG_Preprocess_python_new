import numpy as np
import os
import pandas as pd
from tqdm import tqdm  # 进度条（可选）

# 配置路径
data_root = "Z:/qingzhu/AutoICA_Processed_EEG/DEAP/reordered"  # 替换为你的数据根目录
output_csv = "trial_consistency_scores.csv"   # 输出结果文件

# 获取所有被试和trial信息
subjects = [d for d in os.listdir(data_root) if d.startswith("sub_")]
trials = set()
for sub in subjects:
    sub_dir = os.path.join(data_root, sub)
    for f in os.listdir(sub_dir):
        if f.startswith("eeg_sub") and f.endswith(".npy"):
            trial_id = int(f.split("_")[-1].replace("trial_", "").replace(".npy", ""))
            trials.add(trial_id)
trials = sorted(trials)  # 所有trial的ID列表

# 初始化结果存储
results = []

# 遍历每个trial，计算一致性
for trial in tqdm(trials, desc="Processing trials"):
    trial_data = []
    
    # 加载所有被试的当前trial数据
    for sub in subjects:
        file_path = os.path.join(data_root, sub, f"eeg_sub_{sub.split('_')[1]}_trial_{trial}.npy")
        if os.path.exists(file_path):
            try:
                data = np.load(file_path)  # 形状假设为 (channels, time) 或其他
                trial_data.append(data)
            except:
                print(f'Error read {file_path}')
        else:
            print(f"Warning: Missing file {file_path}")
    
    if not trial_data:
        continue  # 跳过无数据的trial
    
    # 检查所有数据形状是否一致
    shapes = [d.shape for d in trial_data]
    if len(set(shapes)) != 1:
        print(f"Error: Inconsistent shapes for trial {trial}. Skipping.")
        continue
    
    # 计算逐元素乘积之和
    product_sum = np.sum(np.prod(trial_data, axis=0))  # 所有被试数据的element-wise乘积后求和
    results.append({"trial_id": trial, "consistency_score": product_sum})

# 保存结果
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Consistency analysis saved to {output_csv}")