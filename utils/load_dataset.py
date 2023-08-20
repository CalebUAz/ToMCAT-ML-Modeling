import os
import pandas as pd
import numpy as np

def load_dataset_NIRS(path):
    print("-------------------------------------------------------------------------------------------------")
    print("Reading fNIRS affective task data from: {}".format(path))
    print("-------------------------------------------------------------------------------------------------")
    directory = path
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

    columns_to_keep = ["S4-D2_HbO", "S4-D2_HbR", "S4-D5_HbO", "S4-D5_HbR",
                       "S7-D5_HbO", "S7-D5_HbR", "S7-D7_HbR", "S7-D7_HbO", 
                       "S1-D1_HbO", "S1-D1_HbR", "S1-D2_HbO", "S1-D2_HbR", 
                       "S4-D4_HbO", "S4-D4_HbR", "S3-D4_HbO", "S3-D4_HbR", 
                       "S3-D1_HbO", "S3-D1_HbR", "S6-D7_HbO", "S6-D7_HbR", 
                       "S6-D4_HbO", "S6-D4_HbR", "arousal_score", "valence_score"]
    count = 0
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            if folder.startswith('exp_'):
                folder_path = os.path.join(root, folder)
                
                for file in os.listdir(folder_path):
                    if file.startswith('affective_individual_'):
                        file_path = os.path.join(folder_path, file)
                        
                        df = pd.read_csv(file_path)
                        imac = os.path.basename(file).split('_')[0]
                        imac_id = imac + '_id'
                        df.drop(columns=['timestamp_unix', 'station'], inplace=True)
                        
                        if 'task_Unnamed: 0' in df.columns:
                            df.drop(columns='task_Unnamed: 0', inplace=True)

                        if imac in df.columns:
                            df.drop(imac, inplace=True)
                            
                        if imac_id in df.columns:
                            df.drop(columns= imac_id, inplace=True)

                        df.loc[df['event_type'].isin(['intermediate_selection','show_cross_screen', 
                                                      'show_image', 'show_rating_screen']), 
                                                    ['arousal_score', 'valence_score']] = np.nan
                        df[['image_path', 'arousal_score', 
                            'valence_score', 'event_type']] = df[['image_path', 'arousal_score', 
                                                                        'valence_score', 'event_type']].fillna(method='bfill')
                        
                        df.drop(columns=['image_path', 'event_type'], inplace=True)
                        df.dropna(inplace=True)
                        #print(df.columns)
                        df.columns = [None] * len(df.columns)
                        
                        if df.shape[1] == 42:
                            dfs.append(df)
                            count += 1
            
    print("-------------------------")
    print("Number of subjects: {}".format(count))
    print("-------------------------")
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.set_axis(headers, axis=1)
    combined_df = combined_df[columns_to_keep]

    return combined_df

def load_dataset_EEG(path):
    directory = path
    dfs = []
    headers = [
    "AFF1h", "F7", "FC5", "C3", "T7", "TP9", "Pz", "P3", "P7", "O1", "O2", "P8", "P4", "TP10", "Cz", "C4", "T8", "FC6", "FCz", "F8", "AFF2h", "GSR", "EKG", "arousal_score", "valence_score"]

    columns_to_keep = ["FC5", "FCz", "FC6", "F7",
                       "F8", "AFF1h", "AFF2h", "arousal_score", "valence_score"]

    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            if folder.startswith('exp_'):
                folder_path = os.path.join(root, folder)
                
                for file in os.listdir(folder_path):
                    if file.endswith('affective_individual_physio_task.csv'):
                        file_path = os.path.join(folder_path, file)
                        
                        df = pd.read_csv(file_path)
                        imac = os.path.basename(file).split('_')[0]
                        df.drop(columns=['unix_time', 'task_time', 'task_monotonic_time', 'task_human_readable_time', 'task_subject_id', 'seconds_since_start', 'human_readable_time', imac, 'task_index', 'experiment_id'], inplace=True)
                        if 'task_Unnamed: 0' in df.columns:
                            df.drop(columns='task_Unnamed: 0', inplace=True)

                        df.loc[df['task_event_type'].isin(['intermediate_selection','show_cross_screen', 'show_image', 'show_rating_screen']), ['task_arousal_score', 'task_valence_score']] = np.nan
                        df[['task_image_path', 'task_arousal_score', 'task_valence_score', 'task_event_type']] = df[['task_image_path', 'task_arousal_score', 'task_valence_score', 'task_event_type']].fillna(method='bfill')
                        df.drop(columns=['task_image_path', 'task_event_type'], inplace=True)
                        df.dropna(inplace=True)
                        df.columns = [None] * len(df.columns)
                        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.set_axis(headers, axis=1)
    combined_df = combined_df[columns_to_keep]

    return combined_df
