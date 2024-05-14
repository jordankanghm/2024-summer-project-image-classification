import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

# Define paths
TRAIN_IMAGES_DIR = '../datasets/train-images'
TRAIN_LABELS_PATH = '../datasets/train-labels/labels.npy'

# Custom Dataset class to use for the Data Loader
class ImageDataset(Dataset):
    def __init__(self, images_dir, labels, transform=None):
        self.images_dir = images_dir
        self.labels = labels
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Load labels
labels = np.load(TRAIN_LABELS_PATH)

# Define a transform for the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create the dataset
dataset = ImageDataset(TRAIN_IMAGES_DIR, labels, transform=transform)

# Split the dataset into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for the training and test sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
