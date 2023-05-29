import os
import re
import argparse
import pandas as pd

def match_files(raw_folder, extracted_folder):
    raw_files = []
    extracted_files = []
    matched_files = {}

    # Get the list of CSV files in the raw experiment folder
    raw_affective_folder = os.path.join(raw_folder, "baseline_tasks", "affective")
    if os.path.isdir(raw_affective_folder):
        raw_files = [os.path.join(raw_affective_folder, f) for f in os.listdir(raw_affective_folder) if f.endswith(".csv")]

    # Get the list of CSV files in the extracted experiment folder
    if os.path.isdir(extracted_folder):
        extracted_files = [os.path.join(extracted_folder, f) for f in os.listdir(extracted_folder) if f.endswith(".csv")]

    # Match the files
    for raw_file in raw_files:
        match = re.search(r'individual_(\d+)_', raw_file)  # Find the file number using regular expressions
        if match:
            raw_file_number = match.group(1).lstrip("0")
            for extracted_file in extracted_files:
                extracted_file_number = os.path.splitext(os.path.basename(extracted_file))[0]
                if extracted_file_number.isdigit() and int(extracted_file_number) == int(raw_file_number):
                    matched_files[raw_file] = extracted_file
                    break

    return matched_files

def process_csv(raw_file_path, extracted_file_path):
    raw_affective = pd.read_csv(raw_file_path, delimiter=';')
    extracted_affective = pd.read_csv(extracted_file_path)

    # print("Raw Affective Columns:", raw_affective.columns)
    # print("Extracted Affective Columns:", extracted_affective.columns)
    
    # Perform the operations on the dataframes
    extracted_affective.drop("Unnamed: 0", axis=1, inplace=True)
    raw_affective.fillna(method='bfill', inplace=True)
    extracted_affective.fillna(method='bfill', inplace=True)

    # Convert 'human_readable_time' column in extracted_affective to datetime type
    extracted_affective['human_readable_time'] = pd.to_datetime(extracted_affective['human_readable_time'])

    # Perform the merge based on closest matching time
    merged_df = pd.merge_asof(extracted_affective.sort_values('unix_time'),
                              raw_affective.sort_values('time'),
                              left_on='unix_time',
                              right_on='time',
                              direction='nearest')
    
    return merged_df

def main():
    parser = argparse.ArgumentParser(description="Match files between raw and extracted experiment folders.")
    parser.add_argument("--raw-folder", required=True, help="Path to the raw experiment folder")
    parser.add_argument("--extracted-folder", required=True, help="Path to the extracted experiment folder")
    args = parser.parse_args()

    raw_folder_path = args.raw_folder
    extracted_folder_path = args.extracted_folder

    # Call the function to match the files
    matched_files = match_files(raw_folder_path, extracted_folder_path)
    
    # Process each matched pair of csv files
    for raw_file, extracted_file in matched_files.items():
        raw_file_path = os.path.join(raw_folder_path, 'baseline_tasks', 'affective', raw_file)
        extracted_file_path = os.path.join(extracted_folder_path, extracted_file)
        merged_df = process_csv(raw_file_path, extracted_file_path)
        print(merged_df)

if __name__ == "__main__":
    main()
