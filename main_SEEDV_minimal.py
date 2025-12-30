import os
import re
import numpy as np
import mne
from scipy.io import savemat
from collections import defaultdict

from scipy.interpolate import interp1d

def linear_resample(data, orig_sfreq, target_sfreq):
    """
    data: (n_channel, n_sample)
    return: (n_channel, new_n_sample)
    """
    n_ch, n_sample = data.shape
    duration = n_sample / orig_sfreq
    new_n_sample = int(round(duration * target_sfreq))

    t_old = np.linspace(0, duration, n_sample, endpoint=False)
    t_new = np.linspace(0, duration, new_n_sample, endpoint=False)

    data_new = np.zeros((n_ch, new_n_sample), dtype=np.float32)

    for ch in range(n_ch):
        f = interp1d(t_old, data[ch], kind="linear", fill_value="extrapolate")
        data_new[ch] = f(t_new)

    return data_new

# ======================= 配置 =======================
input_dir = "C:\EEG_data\EEG_raw_cnt"
output_dir = "C:\EEG_data\EEG_minimal_mat"
target_sfreq = 125
os.makedirs(output_dir, exist_ok=True)

# ======================= trial 时间 =======================
start_seconds = np.array([
    [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
    [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
    [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888]
])

end_seconds = np.array([
    [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359],
    [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817],
    [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]
])

# ======================= 45 个 trial 标签 =======================
trial_labels = np.array([
    4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0,
    2, 1, 3, 0, 4, 2, 1, 3, 0, 4, 2, 1, 3, 0, 4,
    2, 1, 3, 0, 4, 2, 1, 3, 0, 4, 2, 1, 3, 0, 4
])

# ======================= 文件匹配 =======================
pattern = re.compile(r"(\d+)_(\d)_.*\.cnt")
subject_files = defaultdict(dict)

for f in os.listdir(input_dir):
    m = pattern.match(f)
    if m:
        sub, sess = m.groups()
        subject_files[sub][int(sess)] = os.path.join(input_dir, f)

# ======================= 主处理 =======================
for sub, sessions in subject_files.items():
    print(f"\n处理 sub_{sub}")

    raws = []
    sfreq = None

    # ---- 读取 & 拼 session ----
    for sess in [1, 2, 3]:
        raw = mne.io.read_raw_cnt(
            sessions[sess],
            preload=True,
            verbose=False
        )
        raws.append(raw)
        if sfreq is None:
            sfreq = raw.info['sfreq']

    raw_all = mne.concatenate_raws(raws)
    data_all = raw_all.get_data()   # (C, T)

    data_all = linear_resample(
    data_all,
    orig_sfreq=sfreq,
    target_sfreq=target_sfreq
)

    # ---- trial duration ----
    trial_duration = []
    for s in range(3):
        trial_duration.extend(end_seconds[s] - start_seconds[s])
    trial_duration = np.array(trial_duration)[None, :]  # (1, 45)

    # ---- 保存路径 ----
    sub_dir = os.path.join(output_dir, f"sub_{sub}")
    os.makedirs(sub_dir, exist_ok=True)

    # ================= 保存 data.mat =================
    savemat(
        os.path.join(sub_dir, "data.mat"),
        {
            "data": data_all,
            "trial_duration": trial_duration
        }
    )

    # ================= 保存 label.mat =================
    label_dict = {}
    for i, lbl in enumerate(trial_labels):
        label_dict[f"trial_{i+1}"] = np.array([[lbl]])  # (1, 1)

    savemat(
        os.path.join(sub_dir, "label.mat"),
        label_dict
    )

    print(f"  sub_{sub} 完成，45 trials")

print("\n✅ 所有被试处理完成")
