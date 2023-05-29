import os
import re
import argparse

def match_files(raw_folder, extracted_folder):
    raw_files = []
    extracted_files = []
    matched_files = {}

    # Get the list of CSV files in the raw experiment folder
    raw_affective_folder = os.path.join(raw_folder, "baseline_tasks", "affective")
    if os.path.isdir(raw_affective_folder):
        raw_files = [f for f in os.listdir(raw_affective_folder) if f.endswith(".csv")]

    # Get the list of CSV files in the extracted experiment folder
    if os.path.isdir(extracted_folder):
        extracted_files = [f for f in os.listdir(extracted_folder) if f.endswith(".csv")]

    # Match the files
    for raw_file in raw_files:
        match = re.search(r'individual_(\d+)_', raw_file)  # Find the file number using regular expressions
        if match:
            raw_file_number = match.group(1).lstrip("0")
            for extracted_file in extracted_files:
                extracted_file_number = os.path.splitext(extracted_file)[0]
                if extracted_file_number.isdigit() and int(extracted_file_number) == int(raw_file_number):
                    matched_files[raw_file] = extracted_file
                    break

    return matched_files

def main():
    parser = argparse.ArgumentParser(description="Match files between raw and extracted experiment folders.")
    parser.add_argument("--raw-folder", required=True, help="Path to the raw experiment folder")
    parser.add_argument("--extracted-folder", required=True, help="Path to the extracted experiment folder")
    args = parser.parse_args()

    raw_folder_path = args.raw_folder
    extracted_folder_path = args.extracted_folder

    # Call the function to match the files
    matched_files = match_files(raw_folder_path, extracted_folder_path)
    print(matched_files)
    # Print the matched files
    # if matched_files:
    #     print("Matched files:")
    #     for raw_file, extracted_file in matched_files.items():
    #         print(f"Raw File: {raw_file} | Extracted File: {extracted_file}")
    # else:
    #     print("No matching files found.")

if __name__ == "__main__":
    main()
