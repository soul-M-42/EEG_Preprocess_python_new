# PART 1: Read data

import mne
import numpy as np
import scipy.io
from tqdm import tqdm
import os
from mne_icalabel import label_components
from mne.preprocessing import ICA
import pickle
import json
import time
mne.set_log_level("ERROR")

# EMOEEG DATASET
sfreq = 512
n_sub = 70
volt_factor = 1
channel_names = [
    "FP1", "FPZ", "FP2", 
    "AF3", "AF4", 
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", 
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", 
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", 
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", 
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", 
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", 
    "CB1", "O1", "OZ", "O2", "CB2"
]
eeg_names_Twente = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7',
                    'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3',
                    'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2',
                    'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']
eeg_names_Geneva = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3',
                    'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
                    'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
                    'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

# Select eeg channels
# eeg_channel_names = [
# 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3','AF4','AF8', 'F7', 'F5','F3','F1','Fz', 'F2', 'F4', 'F6', 'F8',
# 'FT7', 'FC5', 'FC3', 'FC1','FCz','FC2','FC4', 'FC6', 'FT8', 'T7','C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
# 'TP7', 'CP5', 'CP3', 'CP1','CPz','CP2', 'CP4','CP6', 'TP8', 'P7','P5', 'P3', 'P1', 'Pz','P2', 'P4', 'P6', 'P8',
# 'PO7', 'PO3','POz', 'PO4','PO8', 'O1','Oz','O2', 'F9', 'F10'
# ]
eeg_channel_names = channel_names
new_rate = 125
thresholds = [
    (3, 0.4),
    (30, 0.01)
]
# montage_file_path = 'Z:/qingzhu/EEG_raw/SEED-VII/src/channel_62_pos.locs'
# montage = mne.channels.read_custom_montage(montage_file_path)
montage = mne.channels.make_standard_montage("standard_1020")
save_dir = 'Z:/qingzhu/AutoICA_Processed_EEG/DEAP'

def detect_bad_channels(raw, thresholds):
    data = raw.get_data(picks='eeg')  # 获取 EEG 数据，形状为 (n_channels, n_times)
    sfreq = raw.info['sfreq']         # 采样频率
    total_samples = data.shape[1]     # 总采样点数
    bad_channels = set()              # 使用集合存储坏道以避免重复
    for a, b in thresholds:
        for ch_idx, ch_data in enumerate(data):
            median = np.median(np.abs(ch_data))
            high_values = np.abs(ch_data) > (a * median)
            high_ratio = np.sum(high_values) / total_samples
            if high_ratio > b:
                bad_channels.add(raw.info['ch_names'][ch_idx])
    return list(bad_channels)

def ICA_denoise(eeg):
    eeg = eeg.set_eeg_reference("average")
    n_components = np.floor(len(eeg.ch_names) * 0.78).astype(int)
    ica = ICA(n_components=n_components, max_iter="auto", random_state=42, method='infomax', fit_params=dict(extended=True))
    # print(f'num_of_nan: {np.isnan(eeg.get_data()).sum()}')
    ica.fit(eeg)
    eeg.load_data()
    ic_labels = label_components(eeg, ica, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    ]
    exclude_idx = exclude_idx[:4]
    eeg_ica = eeg.copy()
    ica.apply(eeg_ica, exclude=exclude_idx)
    return len(exclude_idx), eeg_ica

# raws = [[] for _ in range(n_sub)]
# eeg_selected = [[] for _ in range(n_sub)]
# eeg_filted = [[] for _ in range(n_sub)]
# eeg_interpolated = [[] for _ in range(n_sub)]
# eeg_denoised = [[] for _ in range(n_sub)]
# eeg_interpolated_2nd = [[] for _ in range(n_sub)]

# raws.append([] * n_sub)
# eeg_selected.append([] * n_sub)
# eeg_filted.append([] * n_sub)
# eeg_interpolated.append([] * n_sub)
# eeg_denoised.append([] * n_sub)
# eeg_interpolated_2nd.append([] * n_sub)

for sub in tqdm(range(23, n_sub), desc=f'Reading sub:'):
    try:
        raws_sub = []
        valid_trials_sub = []
        data_path = f'Z:/qingzhu/EEG_raw/DEAP/data_original/s{sub+1:02}.bdf'
        raw_bdf = mne.io.read_raw_bdf(data_path)
        # print(raw_bdf.ch_names)
        if(sub <= 22):
            events = mne.find_events(raw_bdf)
        elif(sub <= 27):
            events = mne.find_events(raw_bdf, stim_channel='')
        else:
            events = mne.find_events(raw_bdf, stim_channel='-1')
        # print(events)
        epochs = mne.Epochs(
            raw_bdf, 
            events, 
            event_id=1638144+4,  # 根据 trigger 划分试次
            tmin=0,  # 试次的开始时间
            tmax=60,   # 试次的结束时间
            baseline=(0, 0)
        )
        print(len(epochs.events))
        # eeg_channel_names = eeg_names_Twente if sub < 22 else eeg_names_Geneva
        eeg_channel_names = eeg_names_Twente
    except:
        print(f"Error read sub {sub}")
        continue
    for trial_id in range(40):
        bad_channels_record = set()
        try:
            info = mne.create_info(ch_names=epochs.ch_names, sfreq=sfreq, ch_types='eeg')
            eeg_trial_i = epochs[trial_id].get_data()[0]
            raw_i = mne.io.RawArray(eeg_trial_i * volt_factor, info)
        except:
            print(f'ERROR Sub {sub} Trial {trial_id}')
            continue

        # selecting channel
        try:
            raw_selected = raw_i.copy().pick_channels(eeg_channel_names)
            # eeg_selected[sub].append(raw_selected)
        except:
            print(f"ERROR In sub {sub} trial {trial_id}: selecting channel")
            # eeg_selected[sub].append(None)

        # Filter & downsample eeg data
        try:
            trial_i_downsampled = raw_selected.resample(sfreq=new_rate)
            trial_i_filted = trial_i_downsampled.filter(1, 47, fir_design='firwin')
            # eeg_filted[sub].append(trial_i_filted)
        except:
            print(f"ERROR In sub {sub} trial {trial_id}: Filter & downsample eeg data")
            # eeg_filted[sub].append(None)

        # 1st bad channel interpolation eeg data
        try:
            trial_i_interploated = trial_i_filted.copy()
            bad_channels = detect_bad_channels(trial_i_filted, thresholds)
            # print(f"1st Detected bad channels: {bad_channels}")
            for cha in bad_channels:
                bad_channels_record.add(cha)
            trial_i_interploated.info['bads'] = bad_channels
            trial_i_interploated.set_montage(montage)
            trial_i_interploated.interpolate_bads(reset_bads=True, exclude=['FP1', 'FP2', 'F7', 'F8', 'AF7', 'AF8'])
            # eeg_interpolated[sub].append(trial_i_interploated)
        except:
            print(f"ERROR In sub {sub} trial {trial_id}: 1st bad channel interpolation eeg data")
            # eeg_interpolated[sub].append(None)

        # ICA denoise
        n_bad_component = -1
        # eeg_denoised[sub].append(trial_i_ica)
        try:
            trial_i_ica = trial_i_interploated.copy()
            n_bad_component, trial_i_ica = ICA_denoise(trial_i_ica)
            # eeg_denoised[sub].append(trial_i_ica)
        except:
            print(f"ERROR In sub {sub} trial {trial_id}: ICA denoise")
            # eeg_denoised[sub].append(None)

        # 2nd bad channel interploation
        try:
            trial_i_interpolated = trial_i_ica.copy()
            bad_channels = detect_bad_channels(trial_i_interpolated, thresholds)
            # print(f"2nd Detected bad channels: {bad_channels}")
            for cha in bad_channels:
                bad_channels_record.add(cha)
            trial_i_interpolated.info['bads'] = bad_channels
            trial_i_interploated.set_montage(montage)
            trial_i_interpolated.interpolate_bads(reset_bads=True)
            trial_i_interpolated.set_eeg_reference("average")
            # eeg_interpolated_2nd[sub].append(trial_i_interpolated)
        except:
            print(f"ERROR In sub {sub} trial {trial_id}: 2nd bad channel interploation")
            # eeg_interpolated_2nd[sub].append(None)

        # Save data
        sub_dir = os.path.join(save_dir, f'sub_{sub}')
        if not os.path.exists(sub_dir):  # 如果子路径不存在，创建它
            os.makedirs(sub_dir)
        bad_channels_record = list(bad_channels_record)
        trial_json = {
            "bad_channel":  bad_channels_record,
            "n_bad_component": n_bad_component,
            "valid": True if len(bad_channels_record) < 10 else False
        }
        json_dir = os.path.join(sub_dir, f'eeg_sub_{sub}_trial_{trial_id}.json')
        with open(json_dir, "w", encoding="utf-8") as f:
            json.dump(trial_json, f, ensure_ascii=False, indent=4)
        try:
            data_dir = os.path.join(sub_dir, f'eeg_sub_{sub}_trial_{trial_id}')
            # trial_i_interpolated.plot()
            # time.sleep(30)
            np.save(data_dir, trial_i_interpolated.get_data())
            print(f"eeg_sub_{sub}_trial_{trial_id} saved.")
        except:
            continue
        