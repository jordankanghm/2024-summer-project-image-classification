import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import models
from image_dataset import ImageDataset

PREPROCESSED_DATA_DIR = '../datasets/preprocessed_data'

def test_model(model, test_loader):
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.tolist())
            true_labels.extend(labels.tolist())
    return true_labels, predicted_labels

# Testing the simple model
model = models.SimpleCNN()
model.load_state_dict(torch.load('simple_model_weights.pth'))

# # Testing the complex model
# model = models.ImprovedCNN()
# model.load_state_dict(torch.load('complex_model_weights.pth'))

# Set the model to evaluation mode
model.eval()

test_loader = torch.load(os.path.join(PREPROCESSED_DATA_DIR, 'test_loader.pth'))

# Test the model using the test_loader from data_preprocessing.py
true_labels, predicted_labels = test_model(model, test_loader)

# Calculate and print evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
