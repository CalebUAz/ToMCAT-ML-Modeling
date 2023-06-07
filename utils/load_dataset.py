import pandas as pd
import numpy as np
import os

def load_dataset(path):
    # directory = '/Users/calebjonesshibu/Desktop/tom/derived/draft_2023_06_05_11/nirs/'

    directory = path
    # Create an empty list to store the DataFrames
    dfs = []

    headers = [
    "S1-D1_HbO", "S1-D2_HbO", "S2-D1_HbO", "S2-D3_HbO", "S3-D1_HbO",
    "S3-D3_HbO", "S3-D4_HbO", "S4-D2_HbO", "S4-D4_HbO", "S4-D5_HbO",
    "S5-D3_HbO", "S5-D4_HbO", "S5-D6_HbO", "S6-D4_HbO", "S6-D6_HbO",
    "S6-D7_HbO", "S7-D5_HbO", "S7-D7_HbO", "S8-D6_HbO", "S8-D7_HbO",
    "S1-D1_HbR", "S1-D2_HbR", "S2-D1_HbR", "S2-D3_HbR", "S3-D1_HbR",
    "S3-D3_HbR", "S3-D4_HbR", "S4-D2_HbR", "S4-D4_HbR", "S4-D5_HbR",
    "S5-D3_HbR", "S5-D4_HbR", "S5-D6_HbR", "S6-D4_HbR", "S6-D6_HbR",
    "S6-D7_HbR", "S7-D5_HbR", "S7-D7_HbR", "S8-D6_HbR", "S8-D7_HbR", 
    "arousal_score", "valence_score"
]
    # Iterate over the directories in the specified path
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            if folder.startswith('exp_') and folder != 'exp_2023_04_18_14': 
                folder_path = os.path.join(root, folder)
                
                # Iterate over the files within the exp_* folder
                for file in os.listdir(folder_path):
                    if file.endswith('affective_individual_physio_task.csv'):
                        file_path = os.path.join(folder_path, file)

                        print(file_path)

                        df = pd.read_csv(file_path)
                        imac = os.path.basename(file).split('_')[0]
                        df.drop(columns=['unix_time', 'task_time', 'task_monotonic_time', 'task_human_readable_time', 'task_subject_id', 'seconds_since_start', 'human_readable_time', imac, 'task_index', 'experiment_id'], inplace=True)
                        df.loc[df['task_event_type'] == 'intermediate_selection', ['task_arousal_score', 'task_valence_score']] = np.nan
                        df[['task_image_path', 'task_arousal_score', 'task_valence_score', 'task_event_type']] = df[['task_image_path', 'task_arousal_score', 'task_valence_score', 'task_event_type']].fillna(method='bfill')
                        df.dropna(inplace=True)
                        df.drop(columns=['task_image_path', 'task_event_type'], inplace=True)
                        df.columns = [None] * len(df.columns)
                        
                        df.reset_index(drop=True, inplace=True)
                        # Append the DataFrame to the list
                        dfs.append(df)

    combined_df = pd.concat(dfs) 
    combined_df.set_axis(headers, axis=1)        

    return combined_df