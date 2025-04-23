import os
import shutil
import pandas as pd

# 配置路径
csv_path = "Z:\qingzhu\EEG_raw\DEAP\metadata_csv\participant_ratings.csv"  # 替换为你的CSV文件路径
src_root = "Z:/qingzhu/AutoICA_Processed_EEG/DEAP/origin"  # 原始数据根目录
dst_root = "Z:/qingzhu/AutoICA_Processed_EEG/DEAP/reordered_correct"  # 新目录（会自动创建）

# 读取CSV文件，获取实验顺序映射
df = pd.read_csv(csv_path)

# 遍历所有被试
for participant_id in df["Participant_id"].unique():
    src_dir = os.path.join(src_root, f"sub_{participant_id-1}")
    dst_dir = os.path.join(dst_root, f"sub_{participant_id-1}")
    os.makedirs(dst_dir, exist_ok=True)

    # 按照标准顺序复制文件
    standard_order = df[df["Participant_id"] == participant_id].sort_values("Trial")["Experiment_id"].tolist()
    for trial_id, exp_id in enumerate(standard_order, start=1):
        src_file = os.path.join(src_dir, f"eeg_sub_{participant_id-1}_trial_{trial_id-1}.npy")
        dst_file = os.path.join(dst_dir, f"eeg_sub_{participant_id-1}_trial_{exp_id-1}.npy")
        
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")
        else:
            print(f"Warning: File not found - {src_file}")

print("Reordering complete!")