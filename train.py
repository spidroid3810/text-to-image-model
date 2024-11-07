import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import TextToImageModel, Discriminator
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

# Data transformations and loading
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor()
])

dataset = TextImageDataset('data/dataset.csv', 'data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize models
generator = TextToImageModel()
discriminator = Discriminator()

# Loss function and optimizers
criterion = torch.nn.BCELoss()  # Binary Cross-Entropy Loss for discriminator
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.002)  # Generator optimizer
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)  # Discriminator optimizer

# Training loop
max_len = 8  # Max length for text inputs

for epoch in range(200):  # Train for 200 epochs
    for text, real_images in dataloader:
        # Encode text: Convert each string to a tensor of character codes (padded to max_len)
        text_inputs = [torch.tensor([ord(c) for c in t]) for t in text]
        text_inputs_padded = torch.zeros((len(text), max_len), dtype=torch.long)

        for i, txt in enumerate(text_inputs):
            end = min(len(txt), max_len)
            text_inputs_padded[i, :end] = txt[:end]

        real_images = Variable(real_images)

        # Train Discriminator
        optimizer_D.zero_grad()
        
        # Generate fake images
        fake_images = generator(text_inputs_padded)

        # Real and fake labels
        real_labels = torch.ones(real_images.size(0), 1)
        fake_labels = torch.zeros(fake_images.size(0), 1)

        # Discriminator loss on real images
        outputs_real = discriminator(real_images)
        d_loss_real = criterion(outputs_real, real_labels)

        # Discriminator loss on fake images
        outputs_fake = discriminator(fake_images.detach())  # Detach to avoid backprop through generator
        d_loss_fake = criterion(outputs_fake, fake_labels)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        # Generate fake images again for generator update
        fake_images = generator(text_inputs_padded)

        # Generator loss: We want the discriminator to classify fake images as real
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)  # Label fake images as real

        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/200], D Loss: {d_loss.item():.3f}, G Loss: {g_loss.item():.3f}")

# Step 1: Prune the generator model (remove 20% of weights in both Linear and Conv2d layers)
for module in generator.modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
        prune.l1_unstructured(module, name="weight", amount=0)
        prune.remove(module, 'weight')  # Remove the pruned connections

# Step 2: Convert the generator model to half precision (16-bit)
generator.half()

# Step 3: Save only the generator model weights
torch.save(generator.state_dict(), 'generator_reduced.pth')
