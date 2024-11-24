import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from FINAL_autoencoder import Autoencoder  # Import the Autoencoder class
from FINAL_dataloader import load_cropped_data, load_annotated_data

# Create an instance of the autoencoder and load the weights
print("Creating autoencoder instance...")
autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load("final_trained_autoencoder.pth"))
autoencoder.eval()  # Switch to evaluation mode
print("Autoencoder model loaded and set to evaluation mode.")

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder.to(device)
print(f"Using device: {device}")

# Function to display and save original and reconstructed images with reconstruction error
def plot_original_and_reconstructed_with_error(originals, reconstructions, title, filename):
    num_images = len(originals)
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 5))
    for i in range(num_images):
        # Originals
        axes[0, i].imshow(originals[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")
        
        # Reconstructions
        reconstruction_error = torch.mean((originals[i] - reconstructions[i]) ** 2).item()
        axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Reconstructed\nError: {reconstruction_error:.4f}")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to: {filename}")
    plt.close()

# Function to select random images from a dataset
def get_random_images(dataset, num_images=5):
    indices = random.sample(range(len(dataset)), num_images)  # Select random indices
    return [dataset[i][0] for i in indices]  # Return corresponding images

# Load datasets
print("Loading Cropped data...")
cropped_dataset = load_cropped_data(
    "Outputs/Cropped_images_red.npz", 
    "Outputs/Cropped_labels_red.npz", 
    batch_size=5, 
    max_images=5, 
    device=device
)
print("Cropped data loaded.")

print("Loading Annotated data...")
annotated_dataloader = load_annotated_data(
    "Outputs/Annotated_Images_red.npz", 
    "Outputs/Annotated_Labels_red.npz", 
    device, 
    batch_size=5, 
    max_images=100
)
print("Annotated data loaded.")

# Select 5 random images from the Cropped dataset
print("Selecting 5 random images from the Cropped dataset...")
cropped_images = get_random_images(cropped_dataset, num_images=5)
print("Random images selected from the Cropped dataset.")

# Select 5 random images from the Annotated dataset
print("Selecting 5 random images from the Annotated dataset...")
annotated_images = []
annotated_labels = []

# Convert the dataloader to a list of images and labels
all_images = []
all_labels = []
for batch_images, batch_labels in annotated_dataloader:
    all_images.extend(batch_images)
    all_labels.extend(batch_labels)

# Select 5 random images from all the collected images
indices = random.sample(range(len(all_images)), 5)
annotated_images = [all_images[i] for i in indices]
annotated_labels = [all_labels[i].item() for i in indices]

print("Random images selected from the Annotated dataset.")

# Reconstruct images using the autoencoder
print("Reconstructing selected images...")
with torch.no_grad():
    cropped_images_tensor = torch.stack(cropped_images).to(device)
    cropped_reconstructed = autoencoder(cropped_images_tensor)

    annotated_images_tensor = torch.stack(annotated_images).to(device)
    annotated_reconstructed = autoencoder(annotated_images_tensor)
print("Reconstruction completed.")

# Save the reconstructed images
print("Saving reconstructed images...")
plot_original_and_reconstructed_with_error(
    cropped_images_tensor, cropped_reconstructed,
    title="Cropped (5 random images)",
    filename="cropped_random_reconstruction_with_error.png"
)

plot_original_and_reconstructed_with_error(
    annotated_images_tensor, annotated_reconstructed,
    title="Annotated (5 random images)",
    filename="annotated_random_reconstruction_with_error.png"
)

print("Process completed.")
