import os
import numpy as np
from PIL import Image

def extract_images(filename, output_dir):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    num_images = len(data) // (28 * 28)
    data = data.reshape(num_images, 28, 28)
    for i, image in enumerate(data):
        image_path = os.path.join(output_dir, f'image_{i}.png')
        # Save the image as PNG (you may need to install the Pillow library)
        Image.fromarray(image).save(image_path)

def extract_labels(filename, output_dir):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    labels_path = os.path.join(output_dir, 'labels.npy')
    np.save(labels_path, data)

current_dir = os.path.dirname(__file__)
# Paths to the binary files
train_images_file = os.path.join(current_dir, '..', 'datasets', 'train-images.idx3-ubyte')
train_labels_file = os.path.join(current_dir, '..', 'datasets', 'train-labels.idx1-ubyte')
test_images_file = os.path.join(current_dir, '..', 'datasets', 't10k-images.idx3-ubyte')
test_labels_file = os.path.join(current_dir, '..', 'datasets', 't10k-labels.idx1-ubyte')

# Output directories for extracted data
train_output_dir = os.path.join(current_dir, '..', 'datasets', 'train-images')
test_output_dir = os.path.join(current_dir, '..', 'datasets', 'test-images')
train_labels_output_dir = os.path.join(current_dir, '..', 'datasets', 'train-labels')
test_labels_output_dir = os.path.join(current_dir, '..', 'datasets', 'test-labels')

# Create output directories if they don't exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)
os.makedirs(train_labels_output_dir, exist_ok=True)
os.makedirs(test_labels_output_dir, exist_ok=True)

# Extract training images
extract_images(train_images_file, train_output_dir)

# Extract testing images
extract_images(test_images_file, test_output_dir)

# Extract training labels
extract_labels(train_labels_file, train_labels_output_dir)

# Extract testing labels
extract_labels(test_labels_file, test_labels_output_dir)
