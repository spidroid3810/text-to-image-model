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
        
