import torch
import torch.nn as nn

class TextToImageModel(nn.Module):
    def __init__(self):
        super(TextToImageModel, self).__init__()
        self.text_embedding = nn.Embedding(1000, 256)  # Text embedding layer
        self.fc1 = nn.Linear(256 * 8, 512)  # Fully connected layer (for max_len=8)
        self.fc2 = nn.Linear(512, 1024)  # Second fully connected layer
        self.fc3 = nn.Linear(1024, 256 * 256 * 3)  # Output: 256x256 RGB image

    def forward(self, text_input):
        x = self.text_embedding(text_input)  # Convert text input to embeddings
        x = x.view(x.size(0), -1)  # Flatten embeddings to (batch_size, 256 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation to scale output to [0, 1]
        x = x.view(-1, 3, 256, 256)  # Reshape to image size (batch_size, 3, 256, 256)
        return x

# Discriminator model to classify real and generated images
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # Output 1x1
            nn.Sigmoid()
        )

    def forward(self, image):
        return self.model(image).view(-1)
