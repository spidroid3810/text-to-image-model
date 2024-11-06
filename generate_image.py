import torch
from model import TextToImageModel
from PIL import Image
import numpy as np

# Load the trained model
model = TextToImageModel()
model.load_state_dict(torch.load('generator_model.pth'))
model.eval()

# Text input (you can change this to any text from your dataset)
text_input = "A 3d goku"  # Change this to test other descriptions

# Simple char-to-int encoding (max_len=8)
max_len = 8
text_tensor = torch.tensor([ord(c) for c in text_input[:max_len]])  # Convert chars to integers
text_tensor = text_tensor.unsqueeze(0)  # Add batch dimension

# Generate the image
with torch.no_grad():
    generated_image = model(text_tensor).cpu()

# Convert the output tensor to an image
generated_image = generated_image.squeeze(0).numpy()  # Remove batch dimension

# Normalize the output range to ensure values are between 0 and 1
generated_image = np.clip(generated_image, 0, 1)

# Scale to [0, 255] for image saving
generated_image = (generated_image * 255).astype(np.uint8)

# Convert from (3, 256, 256) to (256, 256, 3) for saving as an RGB image
generated_image = np.transpose(generated_image, (1, 2, 0))  # Convert to HWC format

# Save and show the image
img = Image.fromarray(generated_image, 'RGB')
img.save('generated_image_1024_fixed.jpg')
img.show()
