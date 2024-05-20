import torch.nn as nn

"""
My initial model was simple: Two convolutional layers with a single linear layer, utilising pooling 
for dimension reduction and better pattern recognition, batch normalisation to mitigate gradient 
vanishing/exploding, ReLU for activation, and dropout to minimise overfitting. The model was run for 20
epochs.

The test statistics are shown below.
"""

"""
Simple CNN Model, two convolutional layers and one linear layer, with batch normalisation, pooling, 
and dropout.
Below are the approximate test statistics.

Training time: 1667.53s
Accuracy: 0.9908
Precision: 0.9907
Recall: 0.9907
F1: 0.9907
"""
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.dropout(self.bn1(nn.functional.relu(self.conv1(x)))))
        x = self.pool(self.dropout(self.bn2(nn.functional.relu(self.conv2(x)))))
        x = x.view(-1, 64 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

"""
Although the test statistics were already high for the simple model, I wanted to try and achieve higher
scores. Hence, I created another model which uses three convolutional layers and two linear layers. The 
model utilised pooling, batch normalisation, ReLU, and Dropout, similar to the simple model. Additionally,
a learning rate scheduler is used to improve the convergence of the model. The model was run for 20 epochs.

The test statistics are shown below.
"""

""""
Complex CNN Model, three convolutional layers and two linear layers, with batch normalisation, pooling
and dropout. A learning rate scheduler was also implemented.
Below are the approximate test statistics.

Training time: 3874.99s
Accuracy: 0.9914
Precision: 0.9914
Recall: 0.9913
F1: 0.9914
"""
# Define a more complex CNN model
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 3 * 3)
        x = self.dropout(self.relu(self.bn4(self.fc1(x))))
        x = self.fc2(x)
        return x

    """
    Comparing the two models, the simple model achieves approximately the same scores but in less than
    half the time. For the sake of computational efficiency, I will conduct hyperparameter tuning on the 
    simple model to optimise the performance of the model.
    """