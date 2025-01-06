# PART 1: Read data

import mne
import numpy as np
import scipy.io
from tqdm import tqdm
import os
from mne_icalabel import label_components
from mne.preprocessing import ICA

# AMIGOS DATASET
sfreq = 200
n_sub = 20
volt_factor = 1e-6
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

# Select eeg channels
eeg_channel_names = [
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
new_rate = 125
thresholds = [
    (3, 0.4),
    (30, 0.01)
]
montage_file_path = 'Z:/qingzhu/EEG_raw/SEED-VII/src/channel_62_pos.locs'
montage = mne.channels.read_custom_montage(montage_file_path)
save_dir = 'Z:/qingzhu/AutoICA_Processed_EEG/SEED-VII'

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
    print(f'num_of_nan: {np.isnan(eeg.get_data()).sum()}')
    ica.fit(eeg)
    eeg.load_data()
    ic_labels = label_components(eeg, ica, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    ]
    eeg_ica = eeg.copy()
    ica.apply(eeg_ica, exclude=exclude_idx)
    return eeg_ica

raws = []
eeg_selected = []
eeg_filted = []
eeg_interpolated = []
eeg_denoised = []
eeg_interpolated_2nd = []

raws.append([] * n_sub)
eeg_selected.append([] * n_sub)
eeg_filted.append([] * n_sub)
eeg_interpolated.append([] * n_sub)
eeg_denoised.append([] * n_sub)
eeg_interpolated_2nd.append([] * n_sub)

for sub in tqdm(range(n_sub), desc=f'Reading sub:'):
    raws_sub = []
    valid_trials_sub = []
    data_path = f'Z:/qingzhu/EEG_raw/SEED-VII/EEG_preprocessed/{sub+1}.mat'
    eeg_data_trials = scipy.io.loadmat(data_path)
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
    for trial_id in range(80):
        eeg_trial_i = eeg_data_trials[str(trial_id+1)]
        try:
            raw_i = mne.io.RawArray(eeg_trial_i * volt_factor, info)
            raws[sub].append(raw_i)
        except:
            print(f'ERROR Sub {sub+1} Trial {trial_id}')
            raws_sub.append(None)
            continue

        # selecting channel
        try:
            raw_selected = raw_i.copy().pick_channels(eeg_channel_names)
            eeg_selected[sub].append(raw_selected)
        except:
            eeg_selected[sub].append(None)

        # Filter & downsample eeg data
        try:
            trial_i_downsampled = raw_selected.resample(sfreq=new_rate)
            trial_i_filted = trial_i_downsampled.filter(1, 47, fir_design='firwin')
            eeg_filted[sub].append(trial_i_filted)
        except:
            eeg_filted[sub].append(None)

        # 1st bad channel interpolation eeg data
        try:
            trial_i_interploated = trial_i_filted.copy()
            bad_channels = detect_bad_channels(trial_i_filted, thresholds)
            print(f"Detected bad channels: {bad_channels}")
            trial_i_interploated.info['bads'] = bad_channels
            trial_i_interploated.set_montage(montage)
            trial_i_interploated.interpolate_bads(reset_bads=True, exclude=['Fp1', 'Fp2', 'F7', 'F8', 'AF7', 'AF8'])
            eeg_interpolated[sub].append(trial_i_interploated)
        except:
            eeg_interpolated[sub].append(None)

        # ICA denoise
        try:
            trial_i_ica = trial_i_interploated.copy()
            trial_i_ica = ICA_denoise(trial_i_ica)
            eeg_denoised[sub].append(trial_i_ica)
        except:
            eeg_denoised[sub].append(None)

        # 2nd bad channel interploation
        try:
            trial_i_interpolated = trial_i_ica.copy()
            bad_channels = detect_bad_channels(trial_i_interpolated, thresholds)
            print(f"Detected bad channels: {bad_channels}")
            trial_i_interpolated.info['bads'] = bad_channels
            trial_i_interploated.set_montage(montage)
            trial_i_interpolated.interpolate_bads(reset_bads=True)
            trial_i_interpolated.set_eeg_reference("average")
            eeg_interpolated_2nd[sub].append(trial_i_interpolated)
        except:
            eeg_interpolated_2nd[sub].append(None)

        # Save data
        sub_dir = os.path.join(save_dir, f'sub_{sub}')
        if not os.path.exists(sub_dir):  # 如果子路径不存在，创建它
            os.makedirs(sub_dir)
        try:
            data_dir = os.path.join(sub_dir, f'eeg_sub_{sub}_trial_{trial_id}')
            np.save(data_dir, trial_i_interpolated.get_data())
        except:
            continue
        