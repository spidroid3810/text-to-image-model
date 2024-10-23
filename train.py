Make the model size as per train.py import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from dataset import TextImageDataset
from model import TextToImageModel
# Define the hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001
# Prepare the dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
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
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
# Save the model
torch.save(model.state_dict(), 'model.pth')
