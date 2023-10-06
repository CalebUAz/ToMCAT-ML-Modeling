import os
import pandas as pd
import numpy as np

def load_dataset_NIRS(path):
    print("-------------------------------------------------------------------------------------------------")
    print("Reading fNIRS affective task data from: {}".format(path))
    print("-------------------------------------------------------------------------------------------------")
    
    # Read ignore experimenter file
    ignore_df = pd.read_csv('utils/ignore_experimenter.csv')

    directory = path
    dfs = []
    subject_ids = []
    headers = [
        "S1-D1_HbO", "S1-D2_HbO", "S2-D1_HbO", "S2-D3_HbO", "S3-D1_HbO",
        "S3-D3_HbO", "S3-D4_HbO", "S4-D2_HbO", "S4-D4_HbO", "S4-D5_HbO",
        "S5-D3_HbO", "S5-D4_HbO", "S5-D6_HbO", "S6-D4_HbO", "S6-D6_HbO",
        "S6-D7_HbO", "S7-D5_HbO", "S7-D7_HbO", "S8-D6_HbO", "S8-D7_HbO",
        "S1-D1_HbR", "S1-D2_HbR", "S2-D1_HbR", "S2-D3_HbR", "S3-D1_HbR",
        "S3-D3_HbR", "S3-D4_HbR", "S4-D2_HbR", "S4-D4_HbR", "S4-D5_HbR",
        "S5-D3_HbR", "S5-D4_HbR", "S5-D6_HbR", "S6-D4_HbR", "S6-D6_HbR",
        "S6-D7_HbR", "S7-D5_HbR", "S7-D7_HbR", "S8-D6_HbR", "S8-D7_HbR", "image_path",
        "arousal_score", "valence_score"
    ]

    columns_to_keep = ["S4-D2_HbO", "S4-D2_HbR", "S4-D5_HbO", "S4-D5_HbR",
                       "S7-D5_HbO", "S7-D5_HbR", "S7-D7_HbR", "S7-D7_HbO", 
                       "S1-D1_HbO", "S1-D1_HbR", "S1-D2_HbO", "S1-D2_HbR", 
                       "S4-D4_HbO", "S4-D4_HbR", "S3-D4_HbO", "S3-D4_HbR", 
                       "S3-D1_HbO", "S3-D1_HbR", "S6-D7_HbO", "S6-D7_HbR", 
                       "S6-D4_HbO", "S6-D4_HbR", "image_path", "arousal_score", "valence_score"]
    count = 0
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            if folder.startswith('exp_'):
                folder_path = os.path.join(root, folder)
                
                for file in os.listdir(folder_path):
                    if file.startswith('affective_individual_'):
                        file_path = os.path.join(folder_path, file)

                        # Check if file path contains an ignore combination
                        station = file.split('_')[-1].split('.')[0] # extract station name from the file name
                        ignore_rows = ignore_df[(ignore_df['group_session'] == folder) & (ignore_df['station'] == station)]
                        if not ignore_rows.empty:
                            print(f"Ignoring file {file_path} because experimenter was sitting there")
                            continue # Skip the rest and move to the next file
                        
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
                        
                        df.drop(columns=['event_type'], inplace=True)
                        df.dropna(inplace=True)
                        #print(df.columns)
                        df.columns = [None] * len(df.columns)
                        
                        if df.shape[1] == 43:
                            dfs.append(df)
                            subject_ids.append(folder + '_' + station)
                            count += 1
            
    print("-------------------------")
    print("Number of subjects: {}".format(count))
    print("-------------------------")
    dfs = [df.reset_index() for df in dfs]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop(columns = ['index'])
    combined_df = combined_df.set_axis(headers, axis=1)
    combined_df = combined_df[columns_to_keep]

    # Subject id for train test split logic
    combined_df['subject_id'] = np.concatenate([[subject] * len(df) for subject, df in zip(subject_ids, dfs)])

    combined_df_temp = combined_df.copy()

    # Define the switch
    look_at_mid = False  # Set this to False if you want the first 50 samples

    if look_at_mid:
        # Calculate cumulative counts
        combined_df_temp['cumcount'] = combined_df_temp.groupby(['subject_id', 'image_path']).cumcount()

        # Calculate total counts for each image_path and map it to the DataFrame
        total_counts = combined_df_temp.groupby(['subject_id', 'image_path']).size()
        combined_df_temp['total_count'] = combined_df_temp.set_index(['subject_id', 'image_path']).index.map(total_counts.to_dict())

        # Calculate the start and end of the mid 50 samples for each row
        combined_df_temp['start_mid'] = (combined_df_temp['total_count'] - 50) / 2
        combined_df_temp['end_mid'] = combined_df_temp['start_mid'] + 50

        # Filter rows where cumcount is within the range of mid 50 samples
        mask = (combined_df_temp['cumcount'] >= combined_df_temp['start_mid']) & (combined_df_temp['cumcount'] < combined_df_temp['end_mid'])
        modified_df = combined_df_temp[mask].drop(columns=['cumcount', 'total_count', 'start_mid', 'end_mid'])

    else:
        # Offset and length definitions
        offset = 50
        length = 20
        # Calculate cumulative counts
        combined_df_temp['cumcount'] = combined_df_temp.groupby(['subject_id', 'image_path']).cumcount()

        # Filter rows where cumcount is within the range after offset
        mask = (combined_df_temp['cumcount'] >= offset) & (combined_df_temp['cumcount'] < offset + length)
        modified_df = combined_df_temp[mask].drop(columns='cumcount')

    # Drop the image_path column as per your script
    combined_df = modified_df.drop(columns=['image_path'])
    # combined_df = combined_df.drop(columns=['image_path'])

    return combined_df

def load_dataset_EEG(path):
    print("-------------------------------------------------------------------------------------------------")
    print("Reading EEG affective task data from: {}".format(path))
    print("-------------------------------------------------------------------------------------------------")
    
    # Read ignore experimenter file
    ignore_df = pd.read_csv('utils/ignore_experimenter.csv')
    
    directory = path
    dfs = [] # List of dataframes
    subject_ids = [] # Subject id for train test split logic
    headers = [
    "AFF1h", "F7", "FC5", "C3", "T7", "TP9", "Pz", "P3", "P7", "O1", "O2", "P8", "P4", "TP10", "Cz", "C4", "T8", "FC6", "FCz", "F8", "AFF2h", "GSR", "EKG", "image_path", "arousal_score", "valence_score"]

    drop_headers = ['AFF5h', 'FC1', 'CP5', 'CP1', 'PO9', 'Oz', 'PO10', 'CP6', 'CP2', 'FC2', 'AFF6h']
    
    columns_to_keep = ["FC5", "FCz", "FC6", "F7",
                       "F8", "AFF1h", "AFF2h", "image_path", "arousal_score", "valence_score"]
    count = 0
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            if folder.startswith('exp_'):
                folder_path = os.path.join(root, folder)
                
                for file in os.listdir(folder_path):
                    if file.startswith('affective_individual_'):
                        file_path = os.path.join(folder_path, file)

                        # Check if file path contains an ignore combination
                        station = file.split('_')[-1].split('.')[0] # extract station name from the file name
                        ignore_rows = ignore_df[(ignore_df['group_session'] == folder) & (ignore_df['station'] == station)]
                        if not ignore_rows.empty:
                            print(f"Ignoring file {file_path} because experimenter was sitting there")
                            continue # Skip the rest and move to the next file
                        
                        df = pd.read_csv(file_path)
                        imac = os.path.basename(file).split('_')[0]
                        imac_id = imac + '_id'
                        df.drop(columns=['timestamp_unix', 'station'], inplace=True)

                        cols_to_drop = [col for col in df.columns if any(col.endswith(header) for header in drop_headers)]
                        df.drop(columns=cols_to_drop, inplace=True) #The number of channels are different before 11/2022
                         
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
                        
                        df.drop(columns=['event_type'], inplace=True)
                        df.dropna(inplace=True)
                        df.columns = [None] * len(df.columns)
                        
                        if df.shape[1] == 26:
                            dfs.append(df)
                            subject_ids.append(folder + '_' + station)
                            count += 1
                        #else:
                            #print(df.shape, file_path)
    print("-------------------------")
    print("Number of subjects: {}".format(count))
    print("-------------------------")
    dfs = [df.reset_index() for df in dfs]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop(columns = ['index'])
    combined_df = combined_df.set_axis(headers, axis=1)
    combined_df = combined_df[columns_to_keep]

    # Subject id for train test split logic
    combined_df['subject_id'] = np.concatenate([[subject] * len(df) for subject, df in zip(subject_ids, dfs)])

    combined_df_temp = combined_df.copy()

   # Define the switch
    look_at_mid = False  # Set this to False if you want the first 50 samples

    if look_at_mid:
        # Calculate cumulative counts
        combined_df_temp['cumcount'] = combined_df_temp.groupby(['subject_id', 'image_path']).cumcount()

        # Calculate total counts for each image_path and map it to the DataFrame
        total_counts = combined_df_temp.groupby(['subject_id', 'image_path']).size()
        combined_df_temp['total_count'] = combined_df_temp.set_index(['subject_id', 'image_path']).index.map(total_counts.to_dict())

        # Calculate the start and end of the mid 50 samples for each row
        combined_df_temp['start_mid'] = (combined_df_temp['total_count'] - 50) / 2
        combined_df_temp['end_mid'] = combined_df_temp['start_mid'] + 50

        # Filter rows where cumcount is within the range of mid 50 samples
        mask = (combined_df_temp['cumcount'] >= combined_df_temp['start_mid']) & (combined_df_temp['cumcount'] < combined_df_temp['end_mid'])
        modified_df = combined_df_temp[mask].drop(columns=['cumcount', 'total_count', 'start_mid', 'end_mid'])

    else:
        # Offset and length definitions
        offset = 1000
        length = 1000
        # Calculate cumulative counts
        combined_df_temp['cumcount'] = combined_df_temp.groupby(['subject_id', 'image_path']).cumcount()

        # Filter rows where cumcount is within the range after offset
        mask = (combined_df_temp['cumcount'] >= offset) & (combined_df_temp['cumcount'] < offset + length)
        modified_df = combined_df_temp[mask].drop(columns='cumcount')

    # Drop the image_path column as per your script
    combined_df = modified_df.drop(columns=['image_path'])
    # combined_df = combined_df.drop(columns=['image_path'])

    return combined_df