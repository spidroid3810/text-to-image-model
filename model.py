import torch 
import torch.nn as nn

class TextToImageModel(nn.Module):
    def __init__(self):
        super(TextToImageModel, self).__init__()
        self.text_embedding = nn.Embedding(1000, 256)  # Reduce embedding size
        self.fc1 = nn.Linear(256 * 8, 32)  # Reduce number of neurons
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 256 * 256 * 3)  # Output 1024x1024 RGB image

    def forward(self, text_input):
        x = self.text_embedding(text_input)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = x.view(-1, 3, 1024, 1024)  # Output larger images
        return x
