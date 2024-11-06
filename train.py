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
    transforms.Resize((256,256)),  # Resize to 256x256
    transforms.ToTensor()
])

dataset = TextImageDataset('data/dataset.csv', 'data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the generator and discriminator
generator = TextToImageModel()
discriminator = Discriminator()

# Loss functions and optimizers
criterion_gan = torch.nn.BCELoss()  # GAN loss for real/fake classification
criterion_mse = torch.nn.MSELoss()  # Mean Squared Error for image generation
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# Training loop
max_len = 8  # Max length for text inputs
num_epochs = 200

for epoch in range(num_epochs):
    for text, real_images in dataloader:
        batch_size = real_images.size(0)
        
        # Encode text: Convert each string to a tensor of character codes (padded to max_len)
        text_inputs = [torch.tensor([ord(c) for c in t]) for t in text]
        text_inputs_padded = torch.zeros((len(text), max_len), dtype=torch.long)

        for i, txt in enumerate(text_inputs):
            end = min(len(txt), max_len)
            text_inputs_padded[i, :end] = txt[:end]

        real_images = Variable(real_images)
        
        # Create labels for real and fake images
        real_labels = torch.ones(batch_size)
        fake_labels = torch.zeros(batch_size)

        # Train Discriminator
        optimizer_d.zero_grad()
        
        # Discriminator loss on real images
        outputs_real = discriminator(real_images)
        loss_real = criterion_gan(outputs_real, real_labels)

        # Generate fake images from the generator
        fake_images = generator(text_inputs_padded).detach()  # Detach to avoid backprop through generator
        outputs_fake = discriminator(fake_images)
        loss_fake = criterion_gan(outputs_fake, fake_labels)
        
        # Total discriminator loss
        d_loss = loss_real + loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        
        # Generate fake images and classify them
        fake_images = generator(text_inputs_padded)
        outputs_fake = discriminator(fake_images)
        
        # Generator loss - want discriminator to think generated images are real
        g_gan_loss = criterion_gan(outputs_fake, real_labels)
        
        # Additional loss (e.g., MSE between generated and real images)
        g_mse_loss = criterion_mse(fake_images, real_images)
        g_loss = g_gan_loss + g_mse_loss
        g_loss.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.3f}, G Loss: {g_loss.item():.3f}")

# Save the generator model weights
torch.save(generator.state_dict(), 'generator_model.pth')

# Save the discriminator model weights
torch.save(discriminator.state_dict(), 'discriminator_model.pth')
