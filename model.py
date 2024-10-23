import torch
import torch.nn as nn

class TextToImageModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, latent_size=128, output_size=64*64*3):
        super(TextToImageModel, self).__init__()
        # Define the encoder (for text input)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU()
        )
        
        # Define the decoder (for image generation)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()  # Output values between -1 and 1
        )
    
    def forward(self, x):
        # Encode the input text into a latent representation
        z = self.encoder(x)
        
        # Decode the latent representation to generate an image
        out = self.decoder(z)
        
        # Reshape the output to image format [Batch, Channels, Height, Width]
        return out.view(-1, 3, 64, 64)

# Example usage
input_size = 128  # Embedding dimension of input text
hidden_size = 256  # Size of the hidden layers
latent_size = 128  # Size of the latent space
output_size = 64 * 64 * 3  # Size of the output image (64x64 RGB)

# Initialize the model
model = TextToImageModel(input_size=input_size, hidden_size=hidden_size, latent_size=latent_size, output_size=output_size)

# Example input (batch of text embeddings)
text_input = torch.randn(32, input_size)  # Batch size of 32

# Forward pass to generate images
generated_images = model(text_input)
print(generated_images.shape)  # Should output [32, 3, 64, 64]
