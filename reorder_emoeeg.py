import os
import numpy as np
from shutil import copyfile

order_1_53 = [
            ['joy5', 'ins5', 'joy8', 'fear8', 'sad8', 'dis5', 'neu4', 'neu5', 'neu8', 'ten5', 'ten8', 'joy4', 'dis4', 'fear4', 'sad4', 'ins8', 'ins4', 'ten4', 'dis8', 'fear5', 'sad5'],
['fear8', 'fear5', 'dis4', 'ins8', 'joy8', 'ins4', 'neu4', 'neu8', 'neu5', 'sad4', 'dis8', 'fear4', 'ten5', 'ten8', 'joy4', 'dis5', 'sad8', 'sad5', 'joy5', 'ten4', 'ins5'],
['ten4', 'joy4', 'joy8', 'neu4', 'neu8', 'neu5', 'dis5', 'fear4', 'fear5', 'ten8', 'ten5', 'ins5', 'fear8', 'dis4', 'dis8', 'ins8', 'joy5', 'ins4', 'sad4', 'sad5', 'sad8'],
['fear5', 'dis8', 'dis5', 'joy4', 'ten5', 'ins5', 'neu4', 'neu8', 'neu5', 'sad8', 'fear8', 'sad4', 'ins4', 'ins8', 'joy8', 'fear4', 'sad5', 'dis4', 'ten4', 'joy5', 'ten8'],
['joy8', 'ten4', 'ins5', 'fear5', 'sad5', 'dis4', 'neu4', 'neu5', 'neu8', 'joy4', 'ten8', 'joy5', 'sad4', 'dis8', 'fear8', 'ins4', 'ten5', 'ins8', 'sad8', 'dis5', 'fear4'],
['joy8', 'ins5', 'ins8', 'dis4', 'dis8', 'fear8', 'ten4', 'joy5', 'ten5', 'dis5', 'fear5', 'fear4', 'ten8', 'ins4', 'joy4', 'sad8', 'sad4', 'sad5', 'neu4', 'neu5', 'neu8'],
['joy8', 'ten8', 'joy4', 'fear4', 'sad5', 'dis5', 'ins5', 'ten5', 'ten4', 'dis4', 'sad8', 'dis8', 'ins4', 'ins8', 'joy5', 'sad4', 'fear8', 'fear5', 'neu4', 'neu5', 'neu8'],
['neu4', 'neu5', 'neu8', 'dis8', 'sad4', 'fear5', 'ins4', 'ins5', 'ten5', 'dis4', 'sad8', 'fear4', 'ins8', 'joy4', 'ten8', 'fear8', 'dis5', 'sad5', 'ten4', 'joy8', 'joy5'],
['sad5', 'fear4', 'fear8', 'joy4', 'joy8', 'ten5', 'dis8', 'dis5', 'sad4', 'neu4', 'neu8', 'neu5', 'ins8', 'ten8', 'ins4', 'sad8', 'fear5', 'dis4', 'joy5', 'ten4', 'ins5'],
['sad4', 'fear5', 'sad8', 'joy8', 'ten8', 'joy4', 'sad5', 'dis8', 'fear4', 'neu4', 'neu8', 'neu5', 'ten4', 'ten5', 'ins4', 'dis4', 'fear8', 'dis5', 'joy5', 'ins5', 'ins8'],
['joy4', 'ins4', 'joy5', 'fear8', 'dis8', 'sad4', 'ten8', 'ins5', 'ten5', 'sad5', 'sad8', 'fear5', 'ins8', 'ten4', 'joy8', 'neu8', 'neu4', 'neu5', 'fear4', 'dis4', 'dis5'],
['sad8', 'fear5', 'fear8', 'ten8', 'ten5', 'joy8', 'fear4', 'sad4', 'sad5', 'neu4', 'neu8', 'neu5', 'ins8', 'ins4', 'ten4', 'dis5', 'dis8', 'dis4', 'joy5', 'joy4', 'ins5'],
['sad8', 'dis8', 'sad4', 'ten4', 'ten8', 'ins4', 'dis5', 'fear8', 'sad5', 'ten5', 'ins5', 'joy8', 'neu4', 'neu8', 'neu5', 'fear5', 'fear4', 'dis4', 'joy4', 'joy5', 'ins8'],
['ins8', 'ten4', 'ins5', 'neu4', 'neu8', 'neu5', 'sad5', 'dis4', 'sad4', 'ins4', 'ten8', 'ten5', 'dis8', 'sad8', 'fear8', 'joy5', 'joy4', 'joy8', 'fear4', 'fear5', 'dis5'],
['ins8', 'ten5', 'ten8', 'sad8', 'sad4', 'sad5', 'joy4', 'ins4', 'ins5', 'fear8', 'fear5', 'fear4', 'ten4', 'joy5', 'joy8', 'neu5', 'neu4', 'neu8', 'dis4', 'dis5', 'dis8'],
['fear4', 'dis4', 'fear8', 'ins8', 'joy8', 'ten8', 'dis5', 'sad4', 'dis8', 'ins5', 'ins4', 'joy4', 'neu8', 'neu4', 'neu5', 'fear5', 'sad8', 'sad5', 'joy5', 'ten5', 'ten4'],
['ten5', 'ins4', 'ins8', 'dis8', 'fear4', 'sad5', 'ins5', 'joy8', 'ten4', 'sad8', 'fear8', 'fear5', 'ten8', 'joy5', 'joy4', 'sad4', 'dis5', 'dis4', 'neu5', 'neu4', 'neu8'],
['neu4', 'neu5', 'neu8', 'sad4', 'dis8', 'dis5', 'joy4', 'ten4', 'ten5', 'sad5', 'fear5', 'fear4', 'ins5', 'ins4', 'ten8', 'dis4', 'fear8', 'sad8', 'joy8', 'ins8', 'joy5'],
['joy5', 'ten8', 'ins4', 'fear4', 'dis8', 'sad4', 'ten5', 'joy8', 'joy4', 'sad8', 'dis5', 'fear8', 'neu8', 'neu4', 'neu5', 'ins5', 'ten4', 'ins8', 'fear5', 'dis4', 'sad5'],
['joy5', 'ins8', 'joy4', 'neu4', 'neu5', 'neu8', 'fear4', 'sad4', 'fear8', 'ins5', 'ten4', 'ten5', 'dis4', 'sad8', 'sad5', 'ten8', 'ins4', 'joy8', 'dis5', 'fear5', 'dis8'],
['ten8', 'joy4', 'ins5', 'sad4', 'dis4', 'fear8', 'ins8', 'joy8', 'ins4', 'neu8', 'neu4', 'neu5', 'sad5', 'sad8', 'fear5', 'ten5', 'joy5', 'ten4', 'fear4', 'dis5', 'dis8'],
['joy5', 'ten8', 'ten4', 'dis4', 'fear4', 'fear5', 'joy8', 'ten5', 'joy4', 'sad5', 'sad8', 'dis8', 'neu5', 'neu8', 'neu4', 'ins4', 'ins5', 'ins8', 'fear8', 'sad4', 'dis5'],
['neu4', 'neu5', 'neu8', 'dis4', 'fear4', 'sad8', 'ins8', 'joy4', 'ten8', 'fear8', 'fear5', 'sad5', 'ten4', 'ins5', 'joy8', 'dis8', 'sad4', 'dis5', 'ten5', 'joy5', 'ins4'],
['joy5', 'ten5', 'ins4', 'fear4', 'sad8', 'sad4', 'ins5', 'ten4', 'ten8', 'sad5', 'fear5', 'fear8', 'ins8', 'joy8', 'joy4', 'dis8', 'dis5', 'dis4', 'neu8', 'neu4', 'neu5'],
['dis8', 'dis5', 'sad4', 'ins8', 'ten4', 'joy8', 'sad8', 'fear4', 'fear8', 'joy5', 'ins4', 'ten8', 'dis4', 'fear5', 'sad5', 'neu8', 'neu5', 'neu4', 'joy4', 'ins5', 'ten5'],
['fear4', 'sad5', 'fear8', 'ten4', 'ins5', 'joy8', 'dis4', 'dis8', 'sad8', 'ins4', 'joy5', 'joy4', 'dis5', 'sad4', 'fear5', 'ins8', 'ten8', 'ten5', 'neu5', 'neu8', 'neu4'],
['dis4', 'dis5', 'fear4', 'ins8', 'ins4', 'joy5', 'sad8', 'fear8', 'sad5', 'ins5', 'joy4', 'ten8', 'neu4', 'neu8', 'neu5', 'fear5', 'sad4', 'dis8', 'ten4', 'ten5', 'joy8'],
['ten4', 'ins5', 'joy4', 'dis5', 'sad5', 'fear4', 'ins8', 'joy8', 'ins4', 'fear5', 'fear8', 'dis8', 'neu5', 'neu8', 'neu4', 'ten8', 'joy5', 'ten5', 'sad4', 'dis4', 'sad8'],
['joy5', 'ten5', 'ins5', 'neu8', 'neu4', 'neu5', 'fear5', 'sad8', 'sad5', 'joy8', 'ten8', 'joy4', 'fear8', 'fear4', 'dis4', 'ten4', 'ins8', 'ins4', 'dis8', 'dis5', 'sad4'],
['sad8', 'dis8', 'dis5', 'joy5', 'ten4', 'joy4', 'sad5', 'fear5', 'fear8', 'ten8', 'ins8', 'ins4', 'sad4', 'fear4', 'dis4', 'joy8', 'ins5', 'ten5', 'neu5', 'neu8', 'neu4'],
['dis4', 'dis8', 'sad4', 'neu5', 'neu4', 'neu8', 'joy5', 'ins8', 'ins4', 'fear4', 'fear8', 'sad8', 'ins5', 'ten8', 'joy4', 'sad5', 'dis5', 'fear5', 'ten4', 'joy8', 'ten5'],
['joy5', 'joy4', 'ten4', 'sad5', 'fear5', 'fear4', 'ins5', 'ten8', 'ins8', 'dis8', 'dis5', 'sad8', 'ten5', 'ins4', 'joy8', 'sad4', 'fear8', 'dis4', 'neu5', 'neu8', 'neu4'],
['sad5', 'dis8', 'dis5', 'ins5', 'ten5', 'ten4', 'dis4', 'fear4', 'fear5', 'ten8', 'ins8', 'joy4', 'neu5', 'neu4', 'neu8', 'fear8', 'sad4', 'sad8', 'joy5', 'joy8', 'ins4'],
['ten5', 'ins5', 'joy4', 'sad4', 'fear5', 'fear4', 'ten8', 'joy8', 'ins8', 'dis8', 'sad5', 'dis5', 'joy5', 'ten4', 'ins4', 'dis4', 'fear8', 'sad8', 'neu4', 'neu8', 'neu5'],
['sad4', 'fear8', 'dis4', 'ins4', 'ins8', 'joy4', 'neu8', 'neu5', 'neu4', 'sad8', 'fear4', 'dis5', 'ten4', 'ten5', 'ten8', 'sad5', 'dis8', 'fear5', 'joy8', 'ins5', 'joy5'],
['joy5', 'joy4', 'joy8', 'dis4', 'dis8', 'fear5', 'neu5', 'neu8', 'neu4', 'ins4', 'ten5', 'ten4', 'dis5', 'sad5', 'fear4', 'ten8', 'ins8', 'ins5', 'sad4', 'sad8', 'fear8'],
['fear4', 'dis5', 'sad5', 'neu5', 'neu4', 'neu8', 'ins8', 'joy8', 'ten5', 'fear5', 'sad4', 'fear8', 'ins4', 'joy4', 'ten8', 'dis4', 'dis8', 'sad8', 'joy5', 'ins5', 'ten4'],
['joy8', 'ten8', 'ins8', 'fear8', 'sad4', 'fear5', 'ten4', 'ten5', 'joy5', 'sad8', 'dis4', 'fear4', 'neu4', 'neu5', 'neu8', 'ins5', 'ins4', 'joy4', 'sad5', 'dis8', 'dis5'],
['ins4', 'ten8', 'joy4', 'neu5', 'neu8', 'neu4', 'dis8', 'fear4', 'sad8', 'ins5', 'joy8', 'ten4', 'dis5', 'dis4', 'fear5', 'ins8', 'ten5', 'joy5', 'fear8', 'sad5', 'sad4'],
['ins4', 'ten4', 'ins5', 'sad5', 'dis5', 'fear4', 'neu5', 'neu8', 'neu4', 'ten5', 'ins8', 'joy4', 'sad8', 'fear5', 'sad4', 'ten8', 'joy5', 'joy8', 'dis8', 'dis4', 'fear8'],
['ins5', 'ten8', 'ins4', 'dis8', 'sad4', 'dis5', 'joy8', 'ten5', 'ins8', 'neu8', 'neu4', 'neu5', 'fear8', 'dis4', 'fear5', 'joy4', 'joy5', 'ten4', 'sad5', 'sad8', 'fear4'],
['ten8', 'ten4', 'joy8', 'dis8', 'sad5', 'sad4', 'joy5', 'ins8', 'ins4', 'neu4', 'neu5', 'neu8', 'fear4', 'dis4', 'fear5', 'ins5', 'ten5', 'joy4', 'dis5', 'fear8', 'sad8'],
['ins5', 'ten5', 'ins4', 'neu5', 'neu8', 'neu4', 'sad4', 'dis4', 'sad5', 'ins8', 'joy8', 'joy4', 'fear8', 'fear4', 'dis8', 'ten8', 'ten4', 'joy5', 'dis5', 'sad8', 'fear5'],
['sad8', 'dis5', 'dis4', 'joy5', 'ins5', 'joy8', 'sad5', 'sad4', 'fear5', 'ten4', 'ten8', 'ins4', 'neu8', 'neu5', 'neu4', 'dis8', 'fear8', 'fear4', 'joy4', 'ten5', 'ins8'],
['ins5', 'joy8', 'ins8', 'fear8', 'fear5', 'sad5', 'joy5', 'ten8', 'ten5', 'neu5', 'neu4', 'neu8', 'dis5', 'dis8', 'sad4', 'ins4', 'ten4', 'joy4', 'sad8', 'dis4', 'fear4'],
['fear5', 'dis5', 'dis8', 'ins5', 'ten5', 'ten8', 'neu8', 'neu4', 'neu5', 'fear8', 'dis4', 'sad4', 'ten4', 'ins8', 'ins4', 'sad5', 'sad8', 'fear4', 'joy8', 'joy4', 'joy5'],
['ins4', 'joy5', 'joy8', 'sad5', 'fear5', 'dis8', 'neu8', 'neu4', 'neu5', 'ins8', 'ten4', 'joy4', 'fear8', 'dis5', 'sad8', 'ins5', 'ten8', 'ten5', 'sad4', 'dis4', 'fear4'],
['joy5', 'ins8', 'ins5', 'dis8', 'dis5', 'fear5', 'ten4', 'ins4', 'joy8', 'dis4', 'fear4', 'sad5', 'ten8', 'ten5', 'joy4', 'fear8', 'sad8', 'sad4', 'neu8', 'neu4', 'neu5'],
['dis4', 'sad5', 'sad4', 'neu4', 'neu8', 'neu5', 'joy4', 'ten5', 'ten8', 'dis8', 'fear8', 'dis5', 'ins4', 'joy8', 'ten4', 'fear4', 'sad8', 'fear5', 'ins8', 'ins5', 'joy5'],
['ten5', 'ins8', 'ins4', 'neu4', 'neu8', 'neu5', 'fear4', 'fear8', 'dis4', 'joy4', 'ten4', 'ins5', 'fear5', 'sad5', 'dis8', 'ten8', 'joy8', 'joy5', 'sad4', 'sad8', 'dis5'],
['ten8', 'joy8', 'ten5', 'dis8', 'fear5', 'dis4', 'joy5', 'ten4', 'ins4', 'fear4', 'sad4', 'dis5', 'neu8', 'neu4', 'neu5', 'ins8', 'ins5', 'joy4', 'sad5', 'fear8', 'sad8'],
['joy4', 'joy5', 'ins8', 'fear5', 'dis5', 'dis8', 'neu5', 'neu4', 'neu8', 'joy8', 'ins5', 'ten5', 'sad5', 'fear4', 'dis4', 'ten4', 'ten8', 'ins4', 'sad8', 'fear8', 'sad4'],
['neu8', 'neu4', 'neu5', 'dis5', 'sad4', 'fear4', 'joy5', 'ins4', 'ten4', 'fear8', 'sad5', 'sad8', 'ten8', 'joy4', 'ten5', 'fear5', 'dis4', 'dis8', 'joy8', 'ins5', 'ins8'],
        ]

def get_reordered_sequence(subject_num, trial_type='imagery'):
    """
    Returns the correct trial sequence for a given subject and trial type.
    """
    # Standard reorder pattern
    standard_reorder = ['sad4', 'sad5', 'sad8', 'dis4', 'dis5', 'dis8', 
                       'fear4', 'fear5', 'fear8', 'neu4', 'neu5', 'neu8', 
                       'joy4', 'joy5', 'joy8', 'ten4', 'ten5', 'ten8', 
                       'ins4', 'ins5', 'ins8']
    
    # Handle special cases
    if subject_num == 54:
        if trial_type == 'imagery':
            return ['ten8', 'ten5', 'ten4', 'fear8', 'fear5', 'fear4', 
                    'dis8', 'dis5', 'dis4', 'joy8', 'joy5', 'joy4', 
                    'sad8', 'sad5', 'sad4', 'neu8', 'neu5', 'neu4', 
                    'ins8', 'ins5', 'ins4']
        else:  # video
            return ['joy8', 'joy5', 'joy4', 'sad8', 'sad5', 'sad4', 
                    'dis8', 'dis5', 'dis4', 'ins8', 'ins5', 'ins4', 
                    'fear8', 'fear5', 'fear4', 'neu8', 'neu5', 'neu4', 
                    'ten8', 'ten5', 'ten4']
    elif subject_num < 54:
        
        return order_1_53[subject_num-1]
        
    else:
        # For sub55-sub60, we don't know the original sequence, so we can't reorder
        raise ValueError(f"Cannot reorder sub{subject_num:02d} without knowing original sequence.")

def reorder_eeg_trials(root_dir, subject_num, trial_type='imagery'):
    """
    Reorders EEG trial files for a subject and saves them in a new directory.
    """
    # Format subject directory name (like 'sub_6')
    sub_dir = f"sub_{subject_num}"
    input_dir = os.path.join(root_dir, sub_dir)
    
    # Create output directory
    output_dir = os.path.join(root_dir, f"{sub_dir}_reordered")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get the correct trial sequence for this subject
        target_sequence = get_reordered_sequence(subject_num, trial_type)
        print()
    except ValueError as e:
        print(f"Skipping {sub_dir}: {str(e)}")
        return
    
    # First collect all existing trial files
    trial_files = []
    for f in os.listdir(input_dir):
        if f.startswith('eeg_') and f.endswith('.npy'):
            # Extract trial number from filename (e.g., 'eeg_sub_6_trial_2.npy' -> 2)
            try:
                trial_num = int(f.split('_')[-1].split('.')[0])
                trial_files.append((trial_num, f))
            except (IndexError, ValueError):
                continue
    
    if not trial_files:
        print(f"No trial files found in {input_dir}")
        return
    
    # Sort by trial number
    trial_files.sort()
    
    # Verify we have the expected number of trials
    if len(trial_files) != len(target_sequence):
        print(f"Warning: Found {len(trial_files)} trials but expected {len(target_sequence)} for {sub_dir}")
    
    # Reorder and save files
    for new_idx, trial_name in enumerate(target_sequence, start=1):
        # Find the original file that corresponds to this trial name
        # This assumes the original files are in the correct order
        # You may need to modify this mapping based on your actual data
        original_idx = target_sequence.index(trial_name) + 1  # +1 because trials start at 1
        
        # Check if we have this original file
        if original_idx > len(trial_files):
            print(f"Warning: Missing trial {trial_name} for {sub_dir}")
            continue
        
        original_num, original_fname = trial_files[original_idx - 1]
        input_path = os.path.join(input_dir, original_fname)
        output_fname = f"eeg_{sub_dir}_trial_{new_idx}.npy"
        output_path = os.path.join(output_dir, output_fname)
        
        # Copy the file (or you could load and save if you need to process the data)
        copyfile(input_path, output_path)
        print(f"Saved {output_fname} (originally trial {original_num} as {trial_name})")

def main():
    # Set your root directory
    root_dir = r"Z:\qingzhu\AutoICA_Processed_EEG\EMOEEG_new"
    
    # Process all subjects
    for subject_num in range(2, 54):  # sub01 to sub60
        # Skip if directory doesn't exist
        sub_dir = os.path.join(root_dir, f"sub_{subject_num}")
        if not os.path.exists(sub_dir):
            continue
        
        print(f"\nProcessing sub_{subject_num}")
        
        # Determine trial type (only affects sub54)
        trial_type = 'imagery'
        if subject_num == 54:
            # Process both imagery and video for sub54
            reorder_eeg_trials(root_dir, subject_num, 'imagery')
            reorder_eeg_trials(root_dir, subject_num, 'video')
            continue
        
        reorder_eeg_trials(root_dir, subject_num, trial_type)


# main()