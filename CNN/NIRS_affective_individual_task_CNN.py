import os
import sys
import time
import pandas as pd
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from utils import save_plot_with_timestamp, sliding_window, load_dataset_NIRS, sliding_window_no_overlap, train_test_split, train_test_split, train_test_split_subject_holdout, sliding_window_get_sub_id

def classify_CNN_Affective_Individual_Task_NIRS(path, hidden_size, num_epochs, batch_size, learning_rate, subject_holdout, window_size):

    # Create the output folder if it doesn't exist
    output_folder = 'output'
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
        except OSError as e:
            print(f"Error creating output folder: {e}")
            return
        
    # Load dataset
    merged_df = load_dataset_NIRS(path)

    if subject_holdout:
        pos = [-3,-2] #Subject hold out has an extra column for subject_id. This is the position of the valence and arousal columns
        print("Using subject holdout for CV")
    else:
        pos = [-2,-1]
        merged_df = merged_df.drop(['subject_id'], axis=1)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess data
    features = merged_df.iloc[:, :pos[0]].values
    arousal_score = LabelEncoder().fit_transform(merged_df.iloc[:, pos[0]] + 2)  # Mapping -2 -> 0, -1 -> 1, 0 -> 2, 1 -> 3, 2 -> 4
    valence_score = LabelEncoder().fit_transform(merged_df.iloc[:, pos[1]] + 2)  # Same mapping for valence_score
    targets = list(zip(arousal_score, valence_score))

    # Get images from sliding window
    look_back = window_size
    # features, valence, arousal = sliding_window(features, valence_score, arousal_score, look_back=look_back)
    features, valence, arousal =  sliding_window_no_overlap(features, valence_score, arousal_score, look_back=look_back)
    targets = list(zip(valence, arousal))

    # Hyperparameters
    input_size = features.shape[1:]
    num_classes = 5  # Classes representing -2, -1, 0, 1, 2
    num_folds = 5

    # Create DataLoaders
    dataset = TensorDataset(torch.tensor(features).float().to(device), torch.tensor(targets).long().to(device))
    if subject_holdout:
        # Use 80% of the subjects for training
        group_split = GroupShuffleSplit(n_splits=num_folds, train_size=0.8, random_state=42)
        groups = merged_df['subject_id'].values
        # Convert string labels to integer labels using LabelEncoder
        encoder = LabelEncoder()
        encoded_groups = encoder.fit_transform(merged_df['subject_id'].values)

        # Get most frequently occurring integer(encoded to subject id) label for each window
        frequent_labels = sliding_window_get_sub_id(encoded_groups, look_back=look_back)

        # If you want the string labels back, decode the integer labels
        groups = encoder.inverse_transform(frequent_labels)
    else:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)


    # Define model
    class CNN(nn.Module):
        def __init__(self, input_shape, num_classes):
            super(CNN, self).__init__()
            
            self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
            self.fc1 = nn.Linear(input_shape[0] * input_shape[1] * 128, 128)  # Adjusted for 2D input
            self.fc_arousal = nn.Linear(128, num_classes)
            self.fc_valence = nn.Linear(128, num_classes)
            
        def forward(self, x):
            x = nn.ReLU()(self.conv1(x))
            x = nn.ReLU()(self.conv2(x))
            x = nn.ReLU()(self.conv3(x))
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = nn.ReLU()(self.fc1(x))
            arousal = self.fc_arousal(x)
            valence = self.fc_valence(x)
            return arousal, valence



    # Initialize model, loss, and optimizer
    model = CNN(input_size, num_classes).to(device)  # Move the model to the GPU

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if subject_holdout:
        fold_losses, fold_accuracies, all_true_arousal, all_pred_arousal, all_true_valence, all_pred_valence = train_test_split_subject_holdout(group_split,groups, targets, dataset, num_folds, num_epochs, batch_size, input_size, model, criterion, optimizer, time, tqdm, Subset, DataLoader, torch, features)
    else:
        fold_losses, fold_accuracies, all_true_arousal, all_pred_arousal, all_true_valence, all_pred_valence = train_test_split(kfold, dataset, num_folds, num_epochs, batch_size, input_size, model, criterion, optimizer, time, tqdm, Subset, DataLoader, torch)

    # Print average accuracy and standard deviation across folds
    arousal_accuracies, valence_accuracies = zip(*fold_accuracies)
    print("Average accuracy for arousal_score:", np.mean(arousal_accuracies))
    print("Standard deviation for arousal_score:", np.std(arousal_accuracies))
    print("Average accuracy for valence_score:", np.mean(valence_accuracies))
    print("Standard deviation for valence_score:", np.std(valence_accuracies))

    # Print the average loss per fold.
    print(f"Average loss per fold: {np.mean(fold_losses)}")
    print(f"Standard deviation of loss per fold: {np.std(fold_losses)}")

    arousal_cm = confusion_matrix(all_true_arousal, all_pred_arousal)
    valence_cm = confusion_matrix(all_true_valence, all_pred_valence)
    
    # Define the class names (assuming -2 to 2 for arousal and valence scores)
    class_names = [-2, -1, 0, 1, 2]

    subject_holdout_str = 'subject_holdout' if subject_holdout else 'regular kfold'

    # Plotting confusion matrix for arousal
    plt.figure(figsize=(20, 14))
    sns.heatmap(arousal_cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues', annot_kws={"size": 16})
    plt.title(f'NIRS-CNN: Confusion Matrix for Arousal\nHidden Size: {hidden_size}, Holdout method: {subject_holdout_str} , Window size: {look_back}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {num_epochs}, Accuracy: {np.mean(arousal_accuracies):.2f}%, std: {np.std(arousal_accuracies):.2f}%, loss: {np.mean(fold_losses):.2f}, std: {np.std(fold_losses):.2f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    save_plot_with_timestamp(plt, 'confusion_matrix_arousal', output_folder)

    # Plotting confusion matrix for valence
    plt.figure(figsize=(20, 14))
    sns.heatmap(valence_cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues', annot_kws={"size": 16})
    plt.title(f'NIRS-CNN: Confusion Matrix for Valence\nHidden Size: {hidden_size}, Holdout method: {subject_holdout_str}, Window size: {look_back}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {num_epochs}, Accuracy: {np.mean(arousal_accuracies):.2f}%, std: {np.std(valence_accuracies):.2f}%, loss: {np.mean(fold_losses):.2f}, std: {np.std(fold_losses):.2f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    save_plot_with_timestamp(plt, 'confusion_matrix_valence', output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post experiment script for xdf to csv file conversion"
    )
    parser.add_argument(
        "--p", required=True, help="Path to the directory with the derived affective task data"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="Hidden size for the CNN model"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for optimizer"
    )

    parser.add_argument(
        "--subject_holdout", type=bool, default=False, help="Use subject holdout for CV"
    )

    parser.add_argument(
        "--window_size", type=int, default=10, help="Use subject holdout for CV"
    )

    args = parser.parse_args()
    path = args.p
    hidden_size = args.hidden_size
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    subject_holdout = args.subject_holdout
    window_size = args.window_size

    sys.exit(classify_CNN_Affective_Individual_Task_NIRS(path, hidden_size, num_epochs, batch_size, learning_rate, subject_holdout, window_size))
