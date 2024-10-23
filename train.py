import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from model import TextToImageModel

# Dataset class to load text and images from CSV
class TextImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Load the CSV file
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

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

        # Convert text to a tensor (e.g., convert characters to ASCII or use a tokenizer)
        text_tensor = torch.tensor([ord(c) for c in text])  # Simple char-to-ASCII encoding

        return text_tensor, image

# Define the hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Data transformations and loading
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the dataset from the CSV file
dataset = TextImageDataset('data/dataset.csv', 'data/images', transform=transform)
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
        
