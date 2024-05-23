import os
from PIL import Image
from torch.utils.data import Dataset

# Custom Dataset class to use for the Data Loader
class ImageDataset(Dataset):
    def __init__(self, images_dir, labels, transform=None):
        self.images_dir = images_dir
        self.labels = labels
        self.transform = transform
        # Sort the images to ensure images and labels are aligned
        self.image_files = sorted([f for f in os.listdir(images_dir)])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
