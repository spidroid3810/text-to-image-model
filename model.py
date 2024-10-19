import torch 
import torch.nn as nn

class TextToImageModel(nn.Module):
    def __init__(self):
        super(TextToImageModel, self).__init__()
        self.text_embedding = nn.Embedding(1000, 16)  # Reduce embedding size
        self.fc1 = nn.Linear(16 * 8, 2)  # Reduce number of neurons
        self.fc2 = nn.Linear(2, 4)
        self.fc3 = nn.Linear(4, 8)
        self.fc4 = nn.Linear(8,16)
        self.fc5 = nn.Linear(16, 1024 * 1024 * 3)  # Output 1024x1024 RGB image

    def forward(self, text_input):
        x = self.text_embedding(text_input)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        x = x.view(-1, 3, 1024, 1024)  # Output larger images
        return x
