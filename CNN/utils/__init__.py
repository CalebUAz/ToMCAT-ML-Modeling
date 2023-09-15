import os 

from .load_dataset import load_dataset_NIRS, load_dataset_EEG
from .save_plot_with_timestamp import save_plot_with_timestamp
from .sliding_window import sliding_window, sliding_window_no_overlap, sliding_window_get_sub_id
from .train_test_split_logic import train_test_split, train_test_split_subject_holdout
from .extract_EEG_features import get_eeg_frequency_band_data

# Function to check if file is empty or doesn't exist
def is_file_empty_or_nonexistent(file_path):
    if not os.path.exists(file_path):
        return True
    if os.stat(file_path).st_size == 0:
        return True
    return False