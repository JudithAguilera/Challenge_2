import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from FINAL_autoencoder import Autoencoder
from FINAL_dataloader import load_annotated_data, load_holdout_data  

# Function to count the number of red pixels (FRED)
def count_red_pixels(image):
    red_pixels = np.sum((image[:, :, 2] > image[:, :, 1]) & (image[:, :, 2] > image[:, :, 0]))
    return red_pixels

# Output path to save files
output_folder = "" # Local

print("loading datasets")
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Annotated and HoldOut data using DataLoader
annotated_loader = load_annotated_data(
    "Outputs/old/Annotated_Images_red.npz",
    "Outputs/old/Annotated_Labels_red.npz",
    device,
    "Outputs/old/Annotated_patient_ids.npz",  # Assuming you have this file
    batch_size=32,
    max_images = 3000
)

holdout_loader = load_holdout_data(
    "Outputs/HoldOut_images_red.npz",
    "Outputs/HoldOut_labels_red.npz",
    "Outputs/HoldOut_patient_ids.npz",  # Assuming you have this file
    device,
    batch_size=32,
    max_images=2000,
    stratified=True
)

# Count the number of each label in Annotated and HoldOut datasets
def count_labels(loader):
    labels = loader.dataset.tensors[1].cpu().numpy()  # Assuming the labels are in the second tensor of the dataset
    unique, counts = np.unique(labels, return_counts=True)
    label_count = dict(zip(unique, counts))
    return label_count

# Print label counts for both datasets
annotated_label_count = count_labels(annotated_loader)
holdout_label_count = count_labels(holdout_loader)

print(f"Annotated dataset label counts: {annotated_label_count}")
print(f"HoldOut dataset label counts: {holdout_label_count}")


print("loading autoencoder")

# Load the trained autoencoder model
autoencoder = Autoencoder() 
autoencoder.load_state_dict(torch.load('final_trained_autoencoder.pth'))
autoencoder.eval()  # Set the model to evaluation mode

# Move the model to GPU if available
autoencoder.to(device)

# Passing through autoencoder for annotated images
print("passing through autoencoder")
reconstructed_annotated_images = []
with torch.no_grad():
    for batch in annotated_loader:
        images, labels = batch
        images = images.to(device)  # Move batch to GPU or CPU
        reconstructed_batch = autoencoder(images)  # Pass batch through model
        reconstructed_batch = reconstructed_batch.cpu().numpy()  # Move back to CPU
        reconstructed_annotated_images.extend(reconstructed_batch)
        # Free up memory
        torch.cuda.empty_cache()  # Clear GPU memory

# Convert reconstructed images back to numpy for pixel comparison
reconstructed_annotated_images_np = np.array(reconstructed_annotated_images)  # Convert back to numpy array

# Calculate the MSE for the annotated dataset
print("Calculating MSE for Annotated Dataset")
mse_annotated = []
for i in range(len(annotated_loader.dataset)):
    original_image = annotated_loader.dataset[i][0].cpu().numpy().astype(float) / 255.0  # Scale to [0, 1]
    reconstructed_image = reconstructed_annotated_images_np[i].astype(float) / 255.0
    mse = np.mean((original_image - reconstructed_image) ** 2)
    mse_annotated.append(mse)

# Calculate statistics of MSE for annotated dataset
mse_mean_annotated = np.mean(mse_annotated)
mse_std_annotated = np.std(mse_annotated)
print(f"MSE Annotated - Mean: {mse_mean_annotated:.4f}, Std: {mse_std_annotated:.4f}")

# Plot histogram of MSE for the annotated dataset
plt.figure()
plt.hist(mse_annotated, bins=10, color='blue', alpha=0.7, edgecolor='black', label='Histogram of MSE')
plt.axvline(x=mse_mean_annotated, color='r', linestyle='-', label=f'Mean: {mse_mean_annotated:.4f}')
plt.axvline(x=mse_mean_annotated + mse_std_annotated, color='g', linestyle='--', label=f'Mean + STD: {mse_mean_annotated + mse_std_annotated:.4f}')
plt.axvline(x=mse_mean_annotated - mse_std_annotated, color='g', linestyle='--', label=f'Mean - STD: {mse_mean_annotated - mse_std_annotated:.4f}')
plt.title('Histogram of MSE for Annotated Dataset')
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# Save the histogram
plt.tight_layout()
plt.savefig("mse_annotated_histogram.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. MSE for Healthy Images in Annotated Dataset
print("Calculating MSE for Healthy Images in Annotated Dataset")
healthy_indices_annotated = [i for i, label in enumerate(annotated_loader.dataset.tensors[1].cpu().numpy()) if label == 0]
healthy_images_annotated = [annotated_loader.dataset[i][0] for i in healthy_indices_annotated]
healthy_reconstructed_images_annotated = reconstructed_annotated_images_np[healthy_indices_annotated]

# Calculate MSE for healthy images in annotated dataset (without repeating autoencoder processing)
mse_healthy_annotated = []
for i in range(len(healthy_images_annotated)):
    original_image = healthy_images_annotated[i].cpu().numpy().astype(float) / 255.0  # Scale to [0, 1]
    reconstructed_image = healthy_reconstructed_images_annotated[i].astype(float) / 255.0
    mse = np.mean((original_image - reconstructed_image) ** 2)
    mse_healthy_annotated.append(mse)

# Plot histogram of MSE for healthy images in annotated dataset
plt.figure()
plt.hist(mse_healthy_annotated, bins=10, color='blue', alpha=0.7, edgecolor='black', label='Histogram of MSE (Healthy Annotated Images)')
mse_mean_healthy_annotated = np.mean(mse_healthy_annotated)
mse_std_healthy_annotated = np.std(mse_healthy_annotated)
plt.axvline(x=mse_mean_healthy_annotated, color='r', linestyle='-', label=f'Mean: {mse_mean_healthy_annotated:.4f}')
plt.axvline(x=mse_mean_healthy_annotated + mse_std_healthy_annotated, color='g', linestyle='--', label=f'Mean + STD: {mse_mean_healthy_annotated + mse_std_healthy_annotated:.4f}')
plt.axvline(x=mse_mean_healthy_annotated - mse_std_healthy_annotated, color='g', linestyle='--', label=f'Mean - STD: {mse_mean_healthy_annotated - mse_std_healthy_annotated:.4f}')
plt.title('Histogram of MSE for Healthy Annotated Images')
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# Save the histogram for healthy annotated images
plt.tight_layout()
plt.savefig("mse_healthy_annotated_histogram.png", dpi=300, bbox_inches='tight')
plt.close()

# Calculate FRED values for annotated images
print("Calculating FRED for Annotated Dataset")
fred_values_annotated = []
for i in range(len(annotated_loader.dataset)):
    original_red_count = count_red_pixels(annotated_loader.dataset[i][0].cpu().numpy())  # Convert to numpy array
    reconstructed_red_count = count_red_pixels(reconstructed_annotated_images_np[i])  # FRED calculation
    
    fred_value = abs(original_red_count - reconstructed_red_count)
    
    fred_values_annotated.append(fred_value)

# Calculate ROC curve and AUC for the annotated dataset
fpr, tpr, thresholds = roc_curve(annotated_loader.dataset.tensors[1].cpu().numpy(), fred_values_annotated)
roc_auc = auc(fpr, tpr)

# Plot and save the ROC curve for annotated dataset
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Annotated)')
plt.legend(loc="lower right")
plt.savefig("roc_curve_annotated.png", dpi=300, bbox_inches='tight')

# Find the optimal threshold for the annotated dataset
cost = (fpr**2 + (tpr - 1)**2)
optimal_threshold_index = np.argmin(cost)
optimal_threshold = thresholds[optimal_threshold_index]

print(f"Optimal threshold (calculated with annotated images): {optimal_threshold}")

# Now classify the annotated images based on the FRED values and optimal threshold
annotated_predictions = [1 if fred >= optimal_threshold else 0 for fred in fred_values_annotated]

# Evaluate performance on annotated dataset
annotated_accuracy = accuracy_score(annotated_loader.dataset.tensors[1].cpu(), annotated_predictions)
print(f"Annotated Accuracy: {annotated_accuracy:.4f}")

# Generate confusion matrix for annotated images
annotated_cm = confusion_matrix(annotated_loader.dataset.tensors[1].cpu(), annotated_predictions)
print("Annotated Confusion Matrix:")
print(annotated_cm)

# Visualize the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(annotated_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Annotated Dataset')
plt.xlabel('Prediction')
plt.ylabel('True Value')

# Save the confusion matrix image
plt.tight_layout()
plt.savefig("confusion_matrix_annotated.png", dpi=300, bbox_inches='tight')
plt.close()

# Generate classification report for annotated images
annotated_report = classification_report(annotated_loader.dataset.tensors[1].cpu(), annotated_predictions)
print("Annotated Classification Report:")
print(annotated_report)

# Save the results in a text file
with open("annotated_results.txt", "w") as file:
    file.write(f"Annotated Accuracy: {annotated_accuracy:.4f}\n\n")
    file.write(f"Optimal Threshold: {optimal_threshold:.4f}\n\n")
    file.write("Annotated Confusion Matrix:\n")
    file.write(np.array2string(annotated_cm, separator=', ') + "\n\n")
    file.write("Annotated Classification Report:\n")
    file.write(annotated_report + "\n")

# Now process the holdout dataset
print("Passing HoldOut dataset through autoencoder")
reconstructed_holdout_images = []
with torch.no_grad():
    for batch in holdout_loader:
        images, labels = batch
        images = images.to(device)
        reconstructed_batch = autoencoder(images)
        reconstructed_batch = reconstructed_batch.cpu().numpy()  # Move back to CPU
        reconstructed_holdout_images.extend(reconstructed_batch)

# Convert reconstructed images back to numpy for pixel comparison
reconstructed_holdout_images_np = np.array(reconstructed_holdout_images)

# Calculate MSE for the HoldOut dataset
print("Calculating MSE for HoldOut Dataset")
mse_holdout = []
for i in range(len(holdout_loader.dataset)):
    original_image = holdout_loader.dataset[i][0].cpu().numpy().astype(float) / 255.0
    reconstructed_image = reconstructed_holdout_images_np[i].astype(float) / 255.0
    mse = np.mean((original_image - reconstructed_image) ** 2)
    mse_holdout.append(mse)

# Calculate statistics of MSE for holdout dataset
mse_mean_holdout = np.mean(mse_holdout)
mse_std_holdout = np.std(mse_holdout)
print(f"MSE HoldOut - Mean: {mse_mean_holdout:.4f}, Std: {mse_std_holdout:.4f}")

# Plot histogram of MSE for the HoldOut dataset
plt.figure()
plt.hist(mse_holdout, bins=10, color='blue', alpha=0.7, edgecolor='black', label='Histogram of MSE')
plt.axvline(x=mse_mean_holdout, color='r', linestyle='-', label=f'Mean: {mse_mean_holdout:.4f}')
plt.axvline(x=mse_mean_holdout + mse_std_holdout, color='g', linestyle='--', label=f'Mean + STD: {mse_mean_holdout + mse_std_holdout:.4f}')
plt.axvline(x=mse_mean_holdout - mse_std_holdout, color='g', linestyle='--', label=f'Mean - STD: {mse_mean_holdout - mse_std_holdout:.4f}')
plt.title('Histogram of MSE for HoldOut Dataset')
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# Save the histogram for the holdout dataset
plt.tight_layout()
plt.savefig("mse_holdout_histogram.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. MSE for Healthy Images in Holdout Dataset
print("Calculating MSE for Healthy Images in HoldOut Dataset")
healthy_indices_holdout = [i for i, label in enumerate(holdout_loader.dataset.tensors[1].cpu().numpy()) if label == 0]
healthy_images_holdout = [holdout_loader.dataset[i][0] for i in healthy_indices_holdout]
healthy_reconstructed_images_holdout = reconstructed_holdout_images_np[healthy_indices_holdout]

# Calculate MSE for healthy images in holdout dataset (without repeating autoencoder processing)
mse_healthy_holdout = []
for i in range(len(healthy_images_holdout)):
    original_image = healthy_images_holdout[i].cpu().numpy().astype(float) / 255.0  # Scale to [0, 1]
    reconstructed_image = healthy_reconstructed_images_holdout[i].astype(float) / 255.0
    mse = np.mean((original_image - reconstructed_image) ** 2)
    mse_healthy_holdout.append(mse)

# Plot histogram of MSE for healthy images in holdout dataset
plt.figure()
plt.hist(mse_healthy_holdout, bins=10, color='blue', alpha=0.7, edgecolor='black', label='Histogram of MSE (Healthy HoldOut Images)')
mse_mean_healthy_holdout = np.mean(mse_healthy_holdout)
mse_std_healthy_holdout = np.std(mse_healthy_holdout)
plt.axvline(x=mse_mean_healthy_holdout, color='r', linestyle='-', label=f'Mean: {mse_mean_healthy_holdout:.4f}')
plt.axvline(x=mse_mean_healthy_holdout + mse_std_healthy_holdout, color='g', linestyle='--', label=f'Mean + STD: {mse_mean_healthy_holdout + mse_std_healthy_holdout:.4f}')
plt.axvline(x=mse_mean_healthy_holdout - mse_std_healthy_holdout, color='g', linestyle='--', label=f'Mean - STD: {mse_mean_healthy_holdout - mse_std_healthy_holdout:.4f}')
plt.title('Histogram of MSE for Healthy HoldOut Images')
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# Save the histogram for healthy holdout images
plt.tight_layout()
plt.savefig("mse_healthy_holdout_histogram.png", dpi=300, bbox_inches='tight')
plt.close()

# Calculate FRED values for holdout images
print("Calculating FRED for HoldOut Dataset")
fred_values_holdout = []
for i in range(len(holdout_loader.dataset)):
    original_red_count = count_red_pixels(holdout_loader.dataset[i][0].cpu().numpy())  # Convert to numpy array
    reconstructed_red_count = count_red_pixels(reconstructed_holdout_images_np[i])  # FRED calculation
    
    fred_value = abs(original_red_count - reconstructed_red_count)
  
    fred_values_holdout.append(fred_value)

'''
# Calculate ROC curve and AUC for holdout dataset
fpr, tpr, thresholds = roc_curve(holdout_loader.dataset.tensors[1].cpu().numpy(), fred_values_holdout)

roc_auc = auc(fpr, tpr)

# Plot and save the ROC curve for holdout dataset
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (HoldOut)')
plt.legend(loc="lower right")
plt.savefig("roc_curve_holdout.png", dpi=300, bbox_inches='tight')

# Find the optimal threshold for the holdout dataset
cost = (fpr**2 + (tpr - 1)**2)
optimal_threshold_index = np.argmin(cost)
optimal_threshold = thresholds[optimal_threshold_index]

print(f"Optimal threshold (calculated with HoldOut images): {optimal_threshold}")
'''

# Now classify the holdout images based on the FRED values and optimal threshold determined later
holdout_predictions = [1 if fred >= optimal_threshold else 0 for fred in fred_values_holdout]

# Evaluate performance on holdout dataset
holdout_accuracy = accuracy_score(holdout_loader.dataset.tensors[1].cpu(), holdout_predictions)
print(f"HoldOut Accuracy: {holdout_accuracy:.4f}")

# Generate confusion matrix for holdout images
holdout_cm = confusion_matrix(holdout_loader.dataset.tensors[1].cpu(), holdout_predictions)
print("HoldOut Confusion Matrix:")
print(holdout_cm)

# Visualize the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(holdout_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - HoldOut Dataset')
plt.xlabel('Prediction')
plt.ylabel('True Value')

# Save the confusion matrix image
plt.tight_layout()
plt.savefig("confusion_matrix_holdout.png", dpi=300, bbox_inches='tight')
plt.close()

# Generate classification report for holdout images
holdout_report = classification_report(holdout_loader.dataset.tensors[1].cpu(), holdout_predictions)
print("HoldOut Classification Report:")
print(holdout_report)

# Save the results for holdout dataset in a text file
with open("holdout_results.txt", "w") as file:
    file.write(f"HoldOut Accuracy: {holdout_accuracy:.4f}\n\n")
    file.write(f"Optimal Threshold: {optimal_threshold:.4f}\n\n")
    file.write("HoldOut Confusion Matrix:\n")
    file.write(np.array2string(holdout_cm, separator=', ') + "\n\n")
    file.write("HoldOut Classification Report:\n")
    file.write(holdout_report + "\n")