import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from image_dataset import ImageDataset
import matplotlib.pyplot as plt

# Define paths
TRAIN_IMAGES_DIR = '../datasets/train-images'
TRAIN_LABELS_PATH = '../datasets/train-labels/labels.npy'
TEST_IMAGES_DIR = '../datasets/test-images'
TEST_LABELS_PATH = '../datasets/test-labels/labels.npy'
PREPROCESSED_DATA_DIR = '../datasets/preprocessed_data'

# Load labels
labels = np.load(TRAIN_LABELS_PATH)
test_labels = np.load(TEST_LABELS_PATH)

# Normalise the images before feeding to the model
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create the dataset
dataset = ImageDataset(TRAIN_IMAGES_DIR, labels, transform=transform)

# Create the test dataset
test_dataset = ImageDataset(TEST_IMAGES_DIR, test_labels, transform=transform)

"""
Original size dataset
"""
# Split the dataset into training and test sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# # Split the dataset into training and test sets with 50% of the entire dataset for quicker training
# total_size = len(dataset)
# train_size = int(total_size * 0.4)
# val_size = int(total_size * 0.1)
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for the training and test sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Create a DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Save the DataLoaders
if not os.path.exists(PREPROCESSED_DATA_DIR):
    os.makedirs(PREPROCESSED_DATA_DIR)

torch.save(train_loader, os.path.join(PREPROCESSED_DATA_DIR, 'train_loader.pth'))
torch.save(val_loader, os.path.join(PREPROCESSED_DATA_DIR, 'val_loader.pth'))
torch.save(test_loader, os.path.join(PREPROCESSED_DATA_DIR, 'val_loader.pth'))

"""
Function to check whether the images and labels in the data loader are correct
"""
# def visualize_batch(data_loader, num_images=10):
#     data_iter = iter(data_loader)
#     images, labels = next(data_iter)
#     images = images.numpy()
#     labels = labels.numpy()
#
#     plt.figure(figsize=(15, 10))
#     for i in range(num_images):
#         plt.subplot(2, 5, i + 1)
#         img = images[i].transpose((1, 2, 0))  # Convert from CHW to HWC format
#         img = img * 0.5 + 0.5  # Unnormalize
#         plt.imshow(img, cmap='gray')
#         plt.title(f"Label: {labels[i]}")
#         plt.axis('off')
#     plt.show()
#
# # Visualize the first 10 training images with labels
# visualize_batch(train_loader, num_images=10)
#
# # Visualize the first 10 test images with labels
# visualize_batch(test_loader, num_images=10)
