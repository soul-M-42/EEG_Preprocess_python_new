import os
import numpy as np
from scipy.io import savemat
import re

# 修改为你的数据所在的根目录
root_dir = 'Z:\qingzhu\AutoICA_Processed_EEG\EMOEEG_Pretrain\original/vid'  # TODO: 修改为实际路径

def natural_key(string_):
    # 把字符串按数字和非数字分割，数字部分转成 int，方便按数字排序
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('(\d+)', string_)]
folder_names = sorted(os.listdir(root_dir), key=natural_key)
print(folder_names)

sampling_rate = 125

vid_len = np.array([98, 120, 56,
           71, 108, 72,
           149, 79, 71,
           65, 106, 114,
           96, 89, 81,
           102, 102, 104,
           109, 63, 110])
for folder_name in folder_names[3:]:
    if folder_name.startswith('sub_') and os.path.isdir(os.path.join(root_dir, folder_name)):
        sub_dir = os.path.join(root_dir, folder_name)
        all_data = []
        n_samples_list = []

        for i_vid, file_name in enumerate(sorted(os.listdir(sub_dir), key=natural_key)):
            print(file_name)
            file_path = os.path.join(sub_dir, file_name)
            data = np.load(file_path)  # shape (64, T)
            # if(data.shape[1]/125 < vid_len[i_vid]):
            #     raise(ValueError(f'{i_vid} min= {data.shape[1]/125}'))
            data = np.concat((np.zeros((62, 100*125)), data), axis=1)
            # print(data)
            data = data[:, -vid_len[i_vid] * 125:]
            # print(data)
            # if i_vid in min_vid_time:
            #     min_vid_time[i_vid] = min(min_vid_time[i_vid], vid_len)
            # else:
            #     min_vid_time[i_vid] = vid_len
            # data = data[:, -30*sampling_rate:]
            # print(data.shape)
            all_data.append(data)
            n_samples_list.append(data.shape[1] / sampling_rate)  # T / 125

        if all_data:
            merged_data_all_cleared = np.concatenate(all_data, axis=1)  # shape (64, T_total)
            merged_n_samples_one = np.array(n_samples_list)  # shape (n_trials,)

            output_path = os.path.join(root_dir, f'{folder_name}_processed.mat')
            savemat(output_path, {
                'merged_data_all_cleared': merged_data_all_cleared,
                'merged_n_samples_one': merged_n_samples_one
            })
            print(f"Saved {output_path}")
