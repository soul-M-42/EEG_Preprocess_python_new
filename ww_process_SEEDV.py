import scipy
from scipy import signal
import os
import pickle
import numpy as np
import mne
import glob
from multiprocessing import Pool
from tqdm import tqdm

# Reference
# https://github.com/wjq-learning/CBraMod/blob/main/preprocessing/preprocessing_SEEDV.py

useless_ch = ['M1', 'M2', 'VEO', 'HEO']
trials_of_sessions = {
    '1': {'start': [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
          'end': [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]},

    '2': {'start': [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
          'end': [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]},

    '3': {'start': [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
          'end': [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]},
}

# 0-Disgust 1-Fear 2-Sad 3-Neutral 4-Happy
labels_of_sessions = {
    '1': [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0, ],
    '2': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0, ],
    '3': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0, ],
}

root_raw = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangwei-240107010004/data/SEED-V/EEG_raw'
root_processed = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangwei-240107010004/BraSigRen/data/eeg/seedv'

files = glob.glob(os.path.join(root_raw, "*.cnt"))
files = sorted(files)
print(files)

trials_split = {
    'train': range(5),
    'val': range(5, 10),
    'test': range(10, 15),
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

def process(file):
    filename = file.split("/")[-1].split(".")[0]
    print(filename)
    raw = mne.io.read_raw_cnt(file, preload=True)
    raw.drop_channels(useless_ch)
    # raw.set_eeg_reference(ref_channels='average')
    raw.resample(200)
    raw.filter(l_freq=0.3, h_freq=75)
    data_matrix = raw.get_data(units='uV')
    session_index = file.split('_')[-2]
    data_trials = [
        data_matrix[:,
        trials_of_sessions[session_index]['start'][j] * 200:trials_of_sessions[session_index]['end'][j] * 200]
        for j in range(15)]
    labels = labels_of_sessions[session_index]
    for mode in trials_split.keys():
        for index in trials_split[mode]:
            data = data_trials[index]
            label = labels[index]
            
            _, t = data.shape
            l_sample = 400
            n_sample = t//l_sample
            for i in range(n_sample):
                sample = data[:, i*l_sample:(i+1)*l_sample]
                out_path = os.path.join(root_processed, mode, filename+'_'+str(index)+'_'+str(i)+".pkl")
                pickle.dump(
                    {"X": sample, "y": label, "ch_names": raw.ch_names},
                    open(out_path, "wb"),
                )

pool = Pool(8)
for _ in tqdm(pool.imap_unordered(process, files), total=len(files)):
    pass
pool.close()
pool.join()          
                