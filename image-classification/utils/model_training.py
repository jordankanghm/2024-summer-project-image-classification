import os
import torch
import torch.nn as nn
import torch.optim as optim
import models
from image_dataset import ImageDataset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

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

"""
Training the simple model
"""
# Instantiate the model, loss function, and optimizer
model = models.SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training function
def train(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        for _, (images, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    end_time = time.time()
    total_time = end_time - start_time
    print("Training complete. Total time taken: {:.2f} seconds".format(total_time))

    # Save the model weights
    torch.save(model.state_dict(), 'simple_model_weights.pth')

# Set the number of epochs
num_epochs = 20

# Train the model
train(model, train_loader, criterion, optimizer, epochs=num_epochs)

"""
Training the improved model
"""
# # Instantiate the model, loss function, and optimizer
# model = models.ImprovedCNN()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
#
# # Initialise a learning rate scheduler
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
#
# # Training function with early stopping
# def train(model, train_loader, criterion, optimizer, scheduler, epochs=20):
#     model.train()
#     start_time = time.time()
#
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for _, (images, labels) in enumerate(train_loader, 1):
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#         avg_loss = running_loss / len(train_loader)
#         print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
#
#         # Update learning rate scheduler
#         scheduler.step(avg_loss)
#
#     end_time = time.time()
#     total_time = end_time - start_time
#     print("Training complete. Total time taken: {:.2f} seconds".format(total_time))
#
#     # Save the model weights
#     torch.save(model.state_dict(), 'complex_model_weights.pth')
#
# # Set the number of epochs and patience for early stopping
# num_epochs = 20
#
# # Train and test the model
# train(model, train_loader, criterion, optimizer, scheduler, num_epochs)


"""
Standard testing function for all models
"""
# Test function
def validate_model(model, test_loader):
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print("Test Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1 Score: {:.4f}".format(f1))
    print("Confusion Matrix:\n", conf_matrix)

validate_model(model, val_loader)
