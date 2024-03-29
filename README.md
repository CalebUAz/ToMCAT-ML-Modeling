# ToMCAT-ML-Modeling

This repository contains scripts that use a separate CNN and LSTM model to classify valence and arousal from EEG and fNIRS labeled datasets. 

## Package Installation
This project requires specific Python packages to be installed. Please follow the instructions below to set up the environment and install the packages you need.
### Environment Setup
1. Make sure you have Python 3.11.3 installed on your system. You can download Python from the official Python website: https://www.python.org/downloads/
2. Create a new virtual environment for this project (optional but recommended). Open a terminal or command prompt and execute the following command:

```bash
python3 -m venv myenv
```
3. Activate the virtual environment. Execute the appropriate command based on your operating system:
* For Windows:
```bash
myenv\Scripts\activate
```
* For Unix or Linux:
```bash
source myenv/bin/activate
```

### Package Installation
4. Install the required packages using pip. In the activated virtual environment, execute the following command:
```bash
pip install -r requirements.txt
```

## Usage:
### fNIRS:
1. Prepare the dataset: Place your fNIRS dataset file in a directory. Note the path to this directory.
2. Execute the script by running the following command:
```bash
python3 NIRS_affective_individual_task_LSTM.py --p  /tomcat/data/derived/drafts/draft_2023_06_05_11/nirs/ --hidden_size 1024 --num_epochs 25 --batch_size 1024
```
Replace `<path_to_dataset_directory>` with the actual path to the directory containing the fNIRS dataset.
Example:
```bash
python3 NIRS_affective_individual_task_LSTM.py --p  /tomcat/data/derived/drafts/draft_2023_06_05_11/eeg/ --hidden_size 1024 --num_epochs 25 --batch_size 1024
```
3. If you wish to pass hyperparameters like hidden_size, num_epochs, batch_size, learning_rate, etc, you can do so by sending those values as a flag. 
4. The script will load the fNIRS affective task  dataset, preprocess the data, train the LSTM model, and evaluate its performance.
5. After completion, the script will display the average accuracy and standard deviation across folds for arousal and valence scores. It will also print the average loss per fold and the standard deviation of loss per fold.
6. Additionally, the script will generate confusion matrix plots for arousal and valence scores and save them in the output folder.

### EEG:
1. Prepare the dataset: Place your EEG dataset file in a directory. Note the path to this directory.
2. Execute the script by running the following command:
```bash
python3 EEG_affective_individual_task_LSTM.py --p <path_to_eeg_dataset_directory>
```
Replace `<path_to_dataset_directory>` with the actual path to the directory containing the EEG dataset.
Example:
```bash
python3 NIRS_affective_individual_task_LSTM.py --p ./data/dataset_directory
```
3. If you wish to pass hyperparameters like hidden_size, num_epochs, batch_size, learning_rate, etc, you can do so by sending those values as a flag. 
4. The script will load the EEG affective task dataset, preprocess the data, train the LSTM model, and evaluate its performance.
5. After completion, the script will display the average accuracy and standard deviation across folds for arousal and valence scores. It will also print the average loss per fold and the standard deviation of loss per fold.
6. Additionally, the script will generate confusion matrix plots for arousal and valence scores and save them in the output folder.
