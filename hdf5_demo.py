import os
import re
import numpy as np
import mne
from scipy.io import savemat
from collections import defaultdict
import h5py
from pathlib import Path

from scipy.interpolate import interp1d

print(np.__version__)
# 1.24.4
print(mne.__version__)
# 0.23.0

output_dir = 'your/output/dir'

ch_names = None
subs = 123456
for sub in range(subs):
    print(sub)
    print(f"处理 sub_{sub}")

    raws = []
    sfreq = None

    trial_segments = []
    # read your trial data......
    for trial_data in trail_eegs:
        # trial_data: [C, T] ndarray
        trial_segments.append(trial_data)

    print(len(trial_segments))
    print(ch_names)
    
    trial_save_path = Path(f"{output_dir}/sub{sub}.h5")
    trial_save_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(trial_save_path, 'w') as f:
        for i_trial, data_trial in enumerate(trial_segments):
            print(data_trial.shape)
            grp_name = f"vid{i_trial}"
            grp = f.create_group(grp_name)
            dset = grp.create_dataset('eeg', data=data_trial)
            dset.attrs['chOrder'] = ch_names
            dset.attrs['rsFreq'] = target_sfreq
            dset.attrs['label'] = trial_labels[i_trial]
            dset.attrs['video_id'] = i_trial
