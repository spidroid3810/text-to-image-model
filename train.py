import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import TextToImageModel

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

# Data transformations and loading
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize to 1024x1024
    transforms.RandomHorizontalFlip(),  # Random flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
    transforms.ToTensor()
])

dataset = TextImageDataset('data/dataset.csv', 'data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Batch size 1

# Initialize the model, loss function, and optimizer
model = TextToImageModel()
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss for image generation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjusted learning rate

# Training loop
max_len = 8  # Max length for text inputs
num_epochs = 10  # Increased epochs

for epoch in range(num_epochs):
    running_loss = 0.0  # Track the loss over the epoch
    
    # Iterate over batches
    for i, (text, images) in enumerate(dataloader):
        # Encode text: Convert each string to a tensor of character codes (padded to max_len)
        text_inputs = [torch.tensor([ord(c) for c in t]) for t in text]
        text_inputs_padded = torch.zeros((len(text), max_len), dtype=torch.long)

        for j, txt in enumerate(text_inputs):
            end = min(len(txt), max_len)
            text_inputs_padded[j, :end] = txt[:end]

        images = Variable(images)

        optimizer.zero_grad()
        outputs = model(text_inputs_padded)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        # Accumulate the loss
        running_loss += loss.item()

    # Calculate average loss for the epoch and print it once after all batches
    average_loss = running_loss / len(dataloader)
    
    # Print the correct epoch number
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.3f}")

# Step 1: Save only the model weights
torch.save(model.state_dict(), 'model_weights_only.pth')

# Optional: If you want to save the full model (weights + architecture), use this:
# torch.save(model, 'full_model.pth')
