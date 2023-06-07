import sys
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
from utils import load_dataset


def train_LSTM_Affective_Individual_Task(path):
    # Load dataset
    merged_df = load_dataset(path)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess data
    features = merged_df.iloc[:, :-2].values
    arousal_score = LabelEncoder().fit_transform(merged_df.iloc[:, -2] + 2)  # Mapping -2 -> 0, -1 -> 1, 0 -> 2, 1 -> 3, 2 -> 4
    valence_score = LabelEncoder().fit_transform(merged_df.iloc[:, -1] + 2)  # Same mapping for valence_score
    targets = list(zip(arousal_score, valence_score))

    # Hyperparameters
    input_size = features.shape[1]
    hidden_size = 64
    num_classes = 5  # Classes representing -2, -1, 0, 1, 2
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001
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
            self.fc = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    # Initialize model, loss, and optimizer
    model = LSTM(input_size, hidden_size, num_classes).to(device)  # Move the model to the GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Perform k-fold cross-validation
    fold_losses = []
    fold_accuracies = []

    for fold, (train_indices, test_indices) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold+1}/{num_folds}")

        # Split data into train and test sets for the current fold
        train_data = Subset(dataset, train_indices)
        test_data = Subset(dataset, test_indices)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Training
        model.train()
        for epoch in range(num_epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.view(-1, 1, input_size)
                targets_arousal = targets[:, 0]
                targets_valence = targets[:, 1]

                outputs = model(inputs)

                loss_arousal = criterion(outputs, targets_arousal)
                loss_valence = criterion(outputs, targets_valence)

                loss = loss_arousal + loss_valence  # Total loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

        fold_losses.append(loss.item())

        # Testing
        model.eval()
        correct_arousal, correct_valence = 0, 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.view(-1, 1, input_size)
                targets_arousal = targets[:, 0]
                targets_valence = targets[:, 1]

                outputs = model(inputs)

                _, predicted_arousal = torch.max(outputs.data, 1)
                _, predicted_valence = torch.max(outputs.data, 1)

                total += targets.size(0)
                correct_arousal += (predicted_arousal == targets_arousal).sum().item()
                correct_valence += (predicted_valence == targets_valence).sum().item()

        accuracy_arousal = 100 * correct_arousal / total
        accuracy_valence = 100 * correct_valence / total
        fold_accuracies.append((accuracy_arousal, accuracy_valence))

    # Print average accuracy and standard deviation across folds
    arousal_accuracies, valence_accuracies = zip(*fold_accuracies)
    print("Average accuracy for arousal_score:", np.mean(arousal_accuracies))
    print("Standard deviation for arousal_score:", np.std(arousal_accuracies))
    print("Average accuracy for valence_score:", np.mean(valence_accuracies))
    print("Standard deviation for valence_score:", np.std(valence_accuracies))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post experiment script for xdf to csv file conversion"
    )
    parser.add_argument(
        "--p", required=True, help="Path to the directory with the derived affective task data"
    )

    arg = parser.parse_args()
    path = arg.p
    sys.exit(train_LSTM_Affective_Individual_Task(path))