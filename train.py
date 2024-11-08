import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import TextToImageModel
import torch.nn.utils.prune as prune
from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("Saad381/SpectraDS")  # Loads the dataset as a Hugging Face dataset object

# Dataset class to load text and images
class TextImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.data = dataset['train']  # Assume we’re using the training split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']  # Access the text column
        image = Image.open(self.data[idx]['file_name'])  # Access the image path column

        # Convert image to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        return text, image

# Data transformations and loading
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor()
])

# Initialize dataset and dataloader
dataset = TextImageDataset(ds, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
model = TextToImageModel()
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss for image generation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
max_len = 8  # Max length for text inputs

for epoch in range(200):  # Train for 200 epochs
    for text, images in dataloader:
        # Encode text: Convert each string to a tensor of character codes (padded to max_len)
        text_inputs = [torch.tensor([ord(c) for c in t]) for t in text]
        text_inputs_padded = torch.zeros((len(text), max_len), dtype=torch.long)

        for i, txt in enumerate(text_inputs):
            end = min(len(txt), max_len)
            text_inputs_padded[i, :end] = txt[:end]

        images = Variable(images)

        optimizer.zero_grad()
        outputs = model(text_inputs_padded)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.3f}")

# Step 1: Prune the model (remove 20% of weights in both Linear and Conv2d layers)
for module in model.modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):  # Prune both Linear and Conv2d layers
        prune.l1_unstructured(module, name="weight", amount=0.2)
        prune.remove(module, 'weight')  # Remove the pruned connections

# Step 2: Convert the model to half precision (16-bit)
model.half()

# Step 3: Save only the model weights
torch.save(model.state_dict(), 'model_reduced.pth')

# Optional: If you don't want to prune or use quantization, you can just save the model like this:
# torch.save(model.state_dict(), 'model_weights_only.pth')
