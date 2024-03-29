import sys
import os
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
from utils import load_dataset_NIRS
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import save_plot_with_timestamp

def classify_LSTM_Affective_Individual_Task_NIRS(path, hidden_size, num_epochs, batch_size, learning_rate):

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

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess data
    features = merged_df.iloc[:, :-2].values
    arousal_score = LabelEncoder().fit_transform(merged_df.iloc[:, -2] + 2)  # Mapping -2 -> 0, -1 -> 1, 0 -> 2, 1 -> 3, 2 -> 4
    valence_score = LabelEncoder().fit_transform(merged_df.iloc[:, -1] + 2)  # Same mapping for valence_score
    targets = list(zip(arousal_score, valence_score))

     # Hyperparameters
    input_size = features.shape[1]
    num_classes = 5  # Classes representing -2, -1, 0, 1, 2
    num_folds = 5

    # Create DataLoaders
    dataset = TensorDataset(torch.tensor(features).float().to(device), torch.tensor(targets).long().to(device))
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Define model
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(LSTM, self).__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc_arousal = nn.Linear(hidden_size, num_classes)
            self.fc_valence = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            arousal = self.fc_arousal(out[:, -1, :])
            valence = self.fc_valence(out[:, -1, :])
            return arousal, valence

    # Initialize model, loss, and optimizer
    model = LSTM(input_size, hidden_size, num_classes).to(device)  # Move the model to the GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Perform k-fold cross-validation
    fold_losses = []
    fold_accuracies = []
    all_true_arousal, all_pred_arousal = [], []
    all_true_valence, all_pred_valence = [], []


    for fold, (train_indices, test_indices) in enumerate(kfold.split(dataset)):
        fold_start_time = time.time() #log the start time of the fold
        print(f"Fold {fold+1}/{num_folds}")

        # Split data into train and test sets for the current fold
        train_data = Subset(dataset, train_indices)
        test_data = Subset(dataset, test_indices)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Print size of the train and test split
        print(f"Size of train data for Fold {fold+1}: {len(train_data)}")
        print(f"Size of test data for Fold {fold+1}: {len(test_data)}")

        # Training
        model.train()
        for epoch in range(num_epochs):
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for i, (inputs, targets) in enumerate(progress_bar):
                inputs = inputs.view(-1, 1, input_size)
                targets_arousal = targets[:, 0]
                targets_valence = targets[:, 1]

                arousal_outputs, valence_outputs = model(inputs)
                loss_arousal = criterion(arousal_outputs, targets_arousal)
                loss_valence = criterion(valence_outputs, targets_valence)

                # loss_arousal = criterion(outputs, targets_arousal)
                # loss_valence = criterion(outputs, targets_valence)

                loss = loss_arousal + loss_valence  # Total loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix(loss=loss.item())

        fold_losses.append(loss.item())
        fold_end_time = time.time()
        fold_elapsed_time = fold_end_time - fold_start_time
        print(f"Time taken for fold {fold+1}: {fold_elapsed_time:.2f} seconds")
        print(f"Loss for Fold {fold+1}: {fold_losses[-1]}")


        # Testing
        model.eval()
        correct_arousal, correct_valence = 0, 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.view(-1, 1, input_size)
                targets_arousal = targets[:, 0]
                targets_valence = targets[:, 1]

                arousal_outputs, valence_outputs = model(inputs)
                _, predicted_arousal = torch.max(arousal_outputs.data, 1)
                _, predicted_valence = torch.max(valence_outputs.data, 1)

                # _, predicted_arousal = torch.max(outputs.data, 1)
                # _, predicted_valence = torch.max(outputs.data, 1)

                all_true_arousal.extend(targets_arousal.cpu().numpy())
                all_pred_arousal.extend(predicted_arousal.cpu().numpy())
                all_true_valence.extend(targets_valence.cpu().numpy())
                all_pred_valence.extend(predicted_valence.cpu().numpy())


                total += targets.size(0)
                correct_arousal += (predicted_arousal == targets_arousal).sum().item()
                correct_valence += (predicted_valence == targets_valence).sum().item()

        accuracy_arousal = 100 * correct_arousal / total
        accuracy_valence = 100 * correct_valence / total
        fold_accuracies.append((accuracy_arousal, accuracy_valence))

        print(f"Unique true arousal classes for Fold {fold+1}: {len(np.unique(targets_arousal.cpu()))}")
        print(f"Unique predicted arousal classes for Fold {fold+1}: {len(np.unique(predicted_arousal.cpu()))}")

        print(f"Unique true valence classes for Fold {fold+1}: {len(np.unique(targets_valence.cpu()))}")
        print(f"Unique predicted valence classes for Fold {fold+1}: {len(np.unique(predicted_valence.cpu()))}")

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


    # arousal_cm = confusion_matrix(targets_arousal.cpu(), predicted_arousal.cpu())
    # valence_cm = confusion_matrix(targets_valence.cpu(), predicted_valence.cpu())
    
    # Define the class names (assuming -2 to 2 for arousal and valence scores)
    class_names = [-2, -1, 0, 1, 2]

    # Plotting confusion matrix for arousal
    plt.figure(figsize=(20, 14))
    sns.heatmap(arousal_cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues', annot_kws={"size": 16})
    plt.title(f'fNIRS: Confusion Matrix for Arousal\nHidden Size: {hidden_size}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {num_epochs}, Accuracy: {np.mean(arousal_accuracies):.2f}%, std: {np.std(arousal_accuracies):.2f}%, loss: {np.mean(fold_losses):.2f}, std: {np.std(fold_losses):.2f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    save_plot_with_timestamp(plt, 'confusion_matrix_arousal', output_folder)

    # Plotting confusion matrix for valence
    plt.figure(figsize=(20, 14))
    sns.heatmap(valence_cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues', annot_kws={"size": 16})
    plt.title(f'fNIRS: Confusion Matrix for Valence\nHidden Size: {hidden_size}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {num_epochs}, Accuracy: {np.mean(arousal_accuracies):.2f}%, std: {np.std(valence_accuracies):.2f}%, loss: {np.mean(fold_losses):.2f}, std: {np.std(fold_losses):.2f}')
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
        "--hidden_size", type=int, default=256, help="Hidden size for the LSTM model"
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

    args = parser.parse_args()
    path = args.p
    hidden_size = args.hidden_size
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    sys.exit(classify_LSTM_Affective_Individual_Task_NIRS(path, hidden_size, num_epochs, batch_size, learning_rate))
    