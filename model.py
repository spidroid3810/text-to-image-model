import torch
import torch.nn as nn

class TextToImageModel(nn.Module):
    def __init__(self):
        super(TextToImageModel, self).__init__()
        self.text_embedding = nn.Embedding(1000, 128)  # Reduced embedding size
        self.fc1 = nn.Linear(128 * 8, 256)  # Reduced number of neurons in fc1
        self.fc2 = nn.Linear(256, 512)  # Reduced fc2 size
        self.fc3 = nn.Linear(512, 64 * 64 * 64)  # Output a smaller feature map (64x64x64)

        # Convolutional layers to upsample to 1024x1024
        self.conv1 = nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1)  # 64x64 -> 128x128
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 128x128 -> 256x256
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # 256x256 -> 512x512
        self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)    # 512x512 -> 1024x1024

    def forward(self, text_input):
        x = self.text_embedding(text_input)  # Convert text input to embeddings
        x = x.view(x.size(0), -1)  # Flatten embeddings to (batch_size, 128 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        # Reshape into a feature map of size (batch_size, 64, 64, 64)
        x = x.view(-1, 64, 64, 64)
        
        # Use convolutional layers to upsample to 1024x1024
        x = torch.relu(self.conv1(x))  # 128x128
        x = torch.relu(self.conv2(x))  # 256x256
        x = torch.relu(self.conv3(x))  # 512x512
        x = torch.sigmoid(self.conv4(x))  # 1024x1024

        return x
