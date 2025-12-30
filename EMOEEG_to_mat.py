import os
import numpy as np
from scipy.io import savemat
import re
from scipy.signal import resample

# 修改为你的数据所在的根目录
root_dir = 'Z:\qingzhu\AutoICA_Processed_EEG\EMOEEG_Pretrain\eeg_npy'  # TODO: 修改为实际路径

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
           109, 63, 110]) # for ima

vid_len = np.array([30] * 21) # for ima

emotion_order = ['joy', 'sad', 'dis', 'ins', 'fear', 'neu', 'ten']
intensity_order = ['8', '5', '4']
def sort_key(filename):
    # 使用正则提取情绪关键词和强度数字
    match = re.search(r'emotion_([a-z]+)(\d)', filename)
    if match:
        emotion, intensity = match.group(1), match.group(2)
        return (emotion_order.index(emotion), intensity_order.index(intensity))
    else:
        return (len(emotion_order), 999)  # fallback if parsing fails
for folder_name in folder_names:
    if folder_name.startswith('sub_') and os.path.isdir(os.path.join(root_dir, folder_name)):
        sub_dir = os.path.join(root_dir, folder_name)
        all_data = []
        n_samples_list = []
        npy_list = sorted([f for f in os.listdir(sub_dir) if (f.endswith('.npy') and 'ima' in f)], key=natural_key)
        print(npy_list)
        npy_list = sorted(npy_list, key=sort_key)
        print(npy_list)
        for i_vid, file_name in enumerate(npy_list):
            print(file_name)
            file_path = os.path.join(sub_dir, file_name)
            data = np.load(file_path)  # shape (64, T)
            print(data.shape[1] / 200)
            data = resample(data, int(data.shape[1] / 200 * sampling_rate), axis=1)
            # if(data.shape[1]/125 < vid_len[i_vid]):
            #     raise(ValueError(f'{i_vid} min= {data.shape[1]/125}'))
            data = np.concat((np.zeros((64, 100*sampling_rate)), data), axis=1)
            # print(data)
            data = data[:, -vid_len[i_vid] * sampling_rate:]
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
