import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd

# Define the TextImageDataset class inside train.py
class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None, max_seq_length=26):
        # Load the CSV file
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.max_seq_length = max_seq_length  # Maximum length for text sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image path and text from CSV
        img_name = self.data.iloc[idx, 0]  # Assumes first column is image path
        text = self.data.iloc[idx, 1]  # Assumes second column is text

        # Load the image
        image = Image.open(f"{self.img_dir}/{img_name}")
        
        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Convert text to a tensor and pad/truncate to max_seq_length
        text_tensor = torch.tensor([ord(c) for c in text[:self.max_seq_length]])  # Truncate text if longer than max_seq_length
        text_tensor = torch.nn.functional.pad(text_tensor, (0, self.max_seq_length - text_tensor.size(0)), value=0)  # Pad text if shorter

        return text_tensor, image

# Define the model (make sure model.py is properly defined and imported)
class TextToImageModel(nn.Module):
    def __init__(self):
        super(TextToImageModel, self).__init__()
        input_size = 26  # Example input size
        hidden_size = 128
        latent_size = 64
        output_size = 64 * 64 * 3  # Output size for a 64x64 RGB image

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out.view(-1, 3, 64, 64)  # Reshape to image dimensions

# Hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001
max_seq_length = 26  # Set the max sequence length for text padding

# Prepare the dataset with CSV file and images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = TextImageDataset('data/dataset.csv', 'data/images', transform=transform, max_seq_length=max_seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the model
model = TextToImageModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for i, (text, image) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, image)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.3f}')

# Save the model
torch.save(model.state_dict(), 'model.pth')
