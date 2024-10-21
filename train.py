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

        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        return text, image

# Data transformations and loading
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize images to 1024x1024
    transforms.ToTensor()
])

dataset = TextImageDataset('data/dataset.csv', 'data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Set batch size to 4

# Initialize the model, loss function, and optimizer
model = TextToImageModel()
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss for image generation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate

# Initialize learning rate scheduler (reduce learning rate every 50 epochs by a factor of 0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# Training loop
max_len = 8  # Max length for text inputs

for epoch in range(300):  # Set number of epochs to 300
    epoch_loss = 0  # Initialize loss for the epoch
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

        epoch_loss += loss.item()  # Accumulate batch loss for the epoch

    # Step the learning rate scheduler at the end of each epoch
    scheduler.step()

    # Print the average loss for the epoch and current learning rate
    print(f"Epoch [{epoch+1}/300], Loss: {epoch_loss/len(dataloader):.3f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# Step 1: Prune the model (remove 20% of weights in both Linear and Conv2d layers)
for module in model.modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):  # Prune both Linear and Conv2d layers
        prune.l1_unstructured(module, name="weight", amount=0.2)
        prune.remove(module, 'weight')  # Remove the pruned connections

# Step 2: Convert the model to half precision (16-bit) to save memory
model.half()

# Step 3: Save only the model weights
torch.save(model.state_dict(), 'model_reduced.safetensors')

# Optional: If you don't want to prune or use quantization, you can just save the model like this:
# torch.save(model.state_dict(), 'model_weights_only.pth')
