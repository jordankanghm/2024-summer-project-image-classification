import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from models import SimpleCNN
from image_dataset import ImageDataset
import optuna

PREPROCESSED_DATA_DIR = '../datasets/preprocessed_data'

# Data augmentations to capture the underlying patterns in the data
train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(28, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_loader = torch.load(os.path.join(PREPROCESSED_DATA_DIR, 'train_loader.pth'))
val_loader = torch.load(os.path.join(PREPROCESSED_DATA_DIR, 'val_loader.pth'))

# Update train_loader with data augmentation
train_loader.dataset.transform = train_transform

def train(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

# Define the validation function
def validate(model, valid_loader, criterion):
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
    return valid_loss / len(valid_loader)

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)

    # Update model and optimizer with suggested hyperparameters
    model = SimpleCNN()
    model.dropout = nn.Dropout(dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training and validation
    num_epochs = 20
    train(model, train_loader, criterion, optimizer, epochs=num_epochs)
    val_loss = validate(model, val_loader, criterion)

    return val_loss

criterion = nn.CrossEntropyLoss()

# Find the optimal hyperparameters
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Get the best trial
best_trial = study.best_trial
print(f'Best trial value: {best_trial.value}')
print(f'Best hyperparameters: {best_trial.params}')

# Train final model with the best hyperparameters
best_lr = best_trial.params['lr']
best_dropout_rate = best_trial.params['dropout_rate']

final_model = SimpleCNN()
final_model.dropout = nn.Dropout(best_dropout_rate)
final_optimizer = optim.Adam(final_model.parameters(), lr=best_lr)

train(final_model, train_loader, criterion, final_optimizer, epochs=20)
torch.save(final_model.state_dict(), 'best_simple_model_weights.pth')