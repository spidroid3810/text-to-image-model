import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import TextToImageModel
import torch.nn.utils.prune as prune

# Dataset class to load text and images
class TextImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 1]  # Access the 'text' column
        img_name = self.data.iloc[idx, 0]  # Access the 'image_path' column
        image = Image.open(f"{self.img_dir}/{img_name}")

        # Convert image to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        return text, image

# Data transformations and loading (without resizing)
transform = transforms.Compose([
    transforms.ToTensor()  # No resize
])

dataset = TextImageDataset('data/dataset.csv', 'data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize the model with the desired output size, e.g., 1024x1024
model = TextToImageModel(output_size=(1024, 1024))  # Model will output 1024x1024 images
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss for image generation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
max_len = 8  # Max length for text inputs

for epoch in range(50):  # Train for 50 epochs
    for text, images in dataloader:
        # Encode text: Convert each string to a tensor of character codes (padded to max_len)
        text_inputs = [torch.tensor([ord(c) for c in t]) for t in text]
        text_inputs_padded = torch.zeros((len(text), max_len), dtype=torch.long)

        for i, txt in enumerate(text_inputs):
            end = min(len(txt), max_len)
            text_inputs_padded[i, :end] = txt[:end]

        images = Variable(images)

        optimizer.zero_grad()
        outputs = model(text_inputs_padded)  # Model generates images of dynamic size (1024x1024 here)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.3f}")

# Save the model
torch.save(model.state_dict(), 'model_reduced.safetensors')
