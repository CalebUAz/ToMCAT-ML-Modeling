import numpy as np
from collections import Counter

def chance_accuracy(train_labels, test_labels):
    most_common_class = Counter(train_labels).most_common(1)[0][0]
    chance_accuracy = sum(1 for y in test_labels if y == most_common_class) / len(test_labels)
    return chance_accuracy

def train_test_split(kfold, dataset, num_folds, num_epochs, batch_size, input_size, model, criterion, optimizer, time, tqdm, Subset, DataLoader, torch):
    # Perform k-fold cross-validation
    fold_losses = []
    fold_accuracies = []
    all_true_arousal, all_pred_arousal = [], []
    all_true_valence, all_pred_valence = [], []

    chance_accuracies_arousal = []
    chance_accuracies_valence = []

    best_loss = float('inf') # Initialize with a high value
    patience_counter = 0 # Counter to keep track of number of epochs with no improvement in loss
    delta=0.9 # Minimum change in the monitored quantity to qualify as an improvement
    patience = 10 # Number of epochs with no improvement after which training will be stopped

    for fold, (train_indices, test_indices) in enumerate(kfold.split(dataset)):
        fold_start_time = time.time() #log the start time of the fold
        print(f"Fold {fold+1}/{num_folds}")

        # Split data into train and test sets for the current fold
        train_data = Subset(dataset, train_indices)
        test_data = Subset(dataset, test_indices)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        # Training
        model.train()
        for epoch in range(num_epochs):
            progress_bar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{num_epochs}")
            for i, (inputs, targets) in enumerate(progress_bar):
                inputs = inputs.view(-1, 1, *input_size)
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

                # Clear cache and delete unnecessary variables
                torch.cuda.empty_cache()
                del inputs, targets

                progress_bar.set_postfix(loss=loss.item())

                 # Early stopping logic
                if loss.item() + delta < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0  # Reset patience counter when there's an improvement
                else:
                    patience_counter += 1  # Increase patience counter if no improvement

                # if patience_counter >= patience:
                #     print("Early Stopping due to no improvement!")
                #     break

        fold_losses.append(loss.item())
        fold_end_time = time.time()
        fold_elapsed_time = fold_end_time - fold_start_time
        print(f"Time taken for fold {fold+1}: {fold_elapsed_time:.2f} seconds")
        print(f"Loss for Fold {fold+1}: {fold_losses[-1]}")


        # Testing
        model.eval()
        correct_arousal, correct_valence = 0, 0
        total = 0

        fold_true_arousal, fold_true_valence = [], []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.view(-1, 1, *input_size)
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

                fold_true_arousal = np.concatenate((fold_true_arousal, targets_arousal.cpu().numpy()))
                fold_true_valence = np.concatenate((fold_true_valence, targets_valence.cpu().numpy()))

                total += targets.size(0)
                correct_arousal += (predicted_arousal == targets_arousal).sum().item()
                correct_valence += (predicted_valence == targets_valence).sum().item()

        accuracy_arousal = 100 * correct_arousal / total
        accuracy_valence = 100 * correct_valence / total
        fold_accuracies.append((accuracy_arousal, accuracy_valence))

        chance_accuracy_arousal = chance_accuracy(fold_true_arousal, all_true_arousal) * 100
        chance_accuracy_valence = chance_accuracy(fold_true_valence, all_true_valence) * 100
        chance_accuracies_arousal.append(chance_accuracy_arousal)
        chance_accuracies_valence.append(chance_accuracy_valence)

    print("\Baseline Accuracies for each fold:")
    for i, (chance_acc_arousal, chance_acc_valence) in enumerate(zip(chance_accuracies_arousal, chance_accuracies_valence)):
        print(f"Fold {i+1}:")
        print(f"  Arousal: {chance_acc_arousal:.2f}%")
        print(f"  Valence: {chance_acc_valence:.2f}%")

    # Calculate and print the average chance accuracies using numpy
    average_chance_accuracy_arousal = np.mean(chance_accuracies_arousal)
    average_chance_accuracy_valence = np.mean(chance_accuracies_valence)

    # Calculate and print the standard deviation for chance accuracies using numpy
    stdev_chance_accuracy_arousal = np.std(chance_accuracies_arousal, ddof=1)
    stdev_chance_accuracy_valence = np.std(chance_accuracies_valence, ddof=1)

    print("\nAverage Baseline accuracy:")
    print(f"  Arousal: {average_chance_accuracy_arousal:.2f}% (± {stdev_chance_accuracy_arousal:.2f}%)")
    print(f"  Valence: {average_chance_accuracy_valence:.2f}% (± {stdev_chance_accuracy_valence:.2f}%)")

    return fold_losses, fold_accuracies, all_true_arousal, all_pred_arousal, all_true_valence, all_pred_valence


def train_test_split_subject_holdout(group_split,groups, targets, dataset, num_folds, num_epochs, batch_size, input_size, model, criterion, optimizer, time, tqdm, Subset, DataLoader, torch, features):

    # Perform k-fold cross-validation
    fold_losses = []
    fold_accuracies = []
    all_true_arousal, all_pred_arousal = [], []
    all_true_valence, all_pred_valence = [], []

    best_loss = float('inf') # Initialize with a high value
    patience_counter = 0 # Counter to keep track of number of epochs with no improvement in loss
    delta=0.1 # Minimum change in the monitored quantity to qualify as an improvement
    patience = 5 # Number of epochs with no improvement after which training will be stopped

    for fold, (train_indices, test_indices) in enumerate(group_split.split(features, targets, groups)):

        # Extract subject_ids for this fold
        train_subjects = np.unique(groups[train_indices])
        test_subjects = np.unique(groups[test_indices])
        
        # Store in the dictionary
        fold_subjects = {}
        fold_subjects[fold] = {
            'train': train_subjects,
            'test': test_subjects
        }

        # Print
        print(f"Fold {fold+1}")
        print(f"Training subjects: {train_subjects}")
        print(f"Testing subjects: {test_subjects}")
        print("-"*40)
        fold_start_time = time.time() #log the start time of the fold
        print(f"Fold {fold+1}/{num_folds}")

        # Split data into train and test sets for the current fold
        train_data = Subset(dataset, train_indices)
        test_data = Subset(dataset, test_indices)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

        # Training
        model.train()
        for epoch in range(num_epochs):
            progress_bar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{num_epochs}")
            for i, (inputs, targets) in enumerate(progress_bar):
                inputs = inputs.view(-1, 1, *input_size)
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

                # Early stopping logic
                if loss.item() + delta < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                # if patience_counter >= patience:
                #     print("Early Stopping due to no improvement!")
                #     break    

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
                inputs = inputs.view(-1, 1, *input_size)
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

    return fold_losses, fold_accuracies, all_true_arousal, all_pred_arousal, all_true_valence, all_pred_valence