import torch
import torch.nn as nn

class TextToImageModel(nn.Module):
    def __init__(self):
        super(TextToImageModel, self).__init__()
        
        # Text embedding layer with smaller embedding size
        self.text_embedding = nn.Embedding(1000, 8)  # Reduced embedding size
        
        # Fully connected layers for text processing
        self.fc1 = nn.Linear(8 * 8, 64)  # Smaller FC layer
        self.fc2 = nn.Linear(64, 128)
        
        # Instead of fully connected layer to output 1024x1024 directly,
        # use a smaller fully connected layer followed by convolution layers
        self.fc3 = nn.Linear(128, 256 * 8 * 8)  # Reduce to 8x8 feature map
        
        # Convolution layers to progressively upsample the output
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # Output: 16x16
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # Output: 32x32
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # Output: 64x64
        self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)    # Output: 128x128
        self.conv5 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)     # Output: 256x256
        self.conv6 = nn.ConvTranspose2d(8, 3, kernel_size=4, stride=4, padding=3)      # Output: 1024x1024 RGB image

    def forward(self, text_input):
        # Text input processing
        x = self.text_embedding(text_input)
        x = x.view(x.size(0), -1)  # Flatten the embedding output
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        # Reshape to 8x8 feature map with 256 channels
        x = x.view(-1, 256, 8, 8)
        
        # Apply convolutional layers to upscale to 1024x1024
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))  # Output scaled between 0 and 1
        
        return x
        
