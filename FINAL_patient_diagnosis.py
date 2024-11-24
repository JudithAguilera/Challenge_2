import os
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from FINAL_autoencoder import Autoencoder
from FINAL_dataloader import load_cropped_data, load_annotated_data, load_holdout_data  # Import dataloader functions

# Load a trained autoencoder model
def load_autoencoder(model_path, device):
    print(f"Loading autoencoder model from {model_path}...")
    model = Autoencoder()  # Initialize the autoencoder class
    state_dict = torch.load(model_path, map_location=device)  # Load the state dictionary
    model.load_state_dict(state_dict)  # Load the weights into the model
    model = model.to(device)  # Move model to the appropriate device
    model.eval()  # Set the model to evaluation mode
    print("Autoencoder model loaded successfully.")
    return model


# Pass images through the autoencoder
def process_with_autoencoder(autoencoder, dataloader, device):
    print("Processing images through the autoencoder...")
    processed_images = []
    with torch.no_grad():  # Disable gradient computation for inference
        for batch_idx, batch in enumerate(dataloader):
            print(f"Processing batch {batch_idx + 1}...")
            images, _ = batch  # Extract images (ignore labels as they are the same)
            images = images.to(device)
            reconstructed = autoencoder(images).cpu()  # Pass through autoencoder
            processed_images.extend(reconstructed.permute(0, 2, 3, 1).numpy())  # Convert to (H, W, C)
    print(f"Processed {len(processed_images)} images.")
    return processed_images

# Load patient diagnoses from CSV
def load_diagnoses(diagnosis_csv):
    print(f"Loading patient diagnoses from {diagnosis_csv}...")
    df = pd.read_csv(diagnosis_csv)
    diagnoses = {row['CODI']: 0 if row['DENSITAT'] == 'negativa' else 1 for _, row in df.iterrows()}
    print(f"Diagnoses loaded for {len(diagnoses)} patients.")
    return diagnoses

# Organize images by patient and assign labels
def organize_by_patient(images, labels, patients, diagnoses):
    print("Organizing images by patient...")
    patient_data = {}
    for img, label, patient in zip(images, labels, patients):
        if patient not in patient_data:
            patient_data[patient] = {'images': [], 'diagnosis': diagnoses.get(patient, -1)}
        patient_data[patient]['images'].append(img)
    print(f"Images organized for {len(patient_data)} patients.")
    return patient_data

# Count red pixels in an image
def count_red_pixels(image):
    return np.sum((image[:, :, 2] > image[:, :, 1]) & (image[:, :, 2] > image[:, :, 0]))

# Classify images based on red pixel threshold
def classify_images(images, red_pixel_threshold):
    print(f"Classifying {len(images)} images based on red pixel threshold...")
    return [1 if count_red_pixels(img) > red_pixel_threshold else 0 for img in images]

# Compute percentage of positive images per patient
def compute_percentages(patient_data, red_pixel_threshold):
    print("Computing percentages of positive images per patient...")
    percentages = []
    for patient_id, data in patient_data.items():
        images = data['images']
        diagnosis = data['diagnosis']
        classifications = classify_images(images, red_pixel_threshold)
        percentage_positive = sum(classifications) / len(classifications) * 100 if images else 0
        percentages.append((patient_id, percentage_positive, diagnosis))
    print("Percentages computed.")
    return percentages

# Compute ROC and AUC metrics
def compute_roc(percentages):
    print("Computing ROC and AUC metrics...")
    predicted_percentages = [p[1] for p in percentages]
    true_labels = [p[2] for p in percentages]
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_percentages)
    roc_auc = auc(fpr, tpr)
    print("ROC and AUC computation completed.")
    return fpr, tpr, thresholds, roc_auc

# Evaluate holdout dataset using the optimal threshold
def evaluate_holdout(holdout_data, optimal_threshold):
    print(f"Evaluating holdout dataset with optimal threshold: {optimal_threshold}...")
    predictions = []
    true_labels = []
    for patient_id, data in holdout_data.items():
        images = data['images']
        diagnosis = data['diagnosis']
        classifications = classify_images(images, optimal_threshold)
        percentage_positive = sum(classifications) / len(classifications) * 100 if images else 0
        predictions.append(1 if percentage_positive >= optimal_threshold else 0)
        true_labels.append(diagnosis)
    accuracy = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    print("Evaluation completed.")
    return accuracy, cm, report

# Initial configuration
diagnosis_csv = 'PatientDiagnosis.csv'
red_pixel_threshold = 3.526  # Red pixel threshold
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder_path = 'final_trained_autoencoder.pth'  

# Load diagnoses
diagnoses = load_diagnoses(diagnosis_csv)

# Load the autoencoder model
autoencoder = load_autoencoder(autoencoder_path, device)

# Load and organize cropped images by patient using the dataloader
cropped_dataset = load_cropped_data('Outputs/Cropped_images_red.npz', 'Outputs/Cropped_labels_red.npz')
cropped_dataloader = torch.utils.data.DataLoader(cropped_dataset, batch_size=64, shuffle=False)

processed_cropped_images = process_with_autoencoder(autoencoder, cropped_dataloader, device)
cropped_labels = np.load('Outputs/Cropped_labels_red.npz')['labels_cropped']
cropped_patients = np.load('Outputs/Cropped_patient_ids.npz')['patient_ids_cropped']

cropped_data = organize_by_patient(processed_cropped_images, cropped_labels, cropped_patients, diagnoses)

# Compute percentages and ROC for cropped dataset
percentages = compute_percentages(cropped_data, red_pixel_threshold)
fpr, tpr, thresholds, roc_auc = compute_roc(percentages)

print(f"Area Under the ROC Curve (AUC): {roc_auc:.2f}")

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Find the optimal threshold
cost = (fpr**2 + (tpr - 1)**2)
optimal_threshold_index = np.argmin(cost)
optimal_threshold = thresholds[optimal_threshold_index]

print(f"Optimal threshold: {optimal_threshold}")

# Load and process holdout dataset
holdout_dataset = load_holdout_data('Outputs/HoldOut_images_red.npz', 'Outputs/HoldOut_labels_red.npz', 'Outputs/HoldOut_patient_ids.npz', device)
holdout_dataloader = torch.utils.data.DataLoader(holdout_dataset, batch_size=64, shuffle=False)

processed_holdout_images = process_with_autoencoder(autoencoder, holdout_dataloader, device)
holdout_labels = np.load('Outputs/HoldOut_labels_red.npz')['labels_HOLDOUT']
holdout_patients = np.load('Outputs/HoldOut_patient_ids.npz')['patient_ids_holdout']

holdout_data = organize_by_patient(processed_holdout_images, holdout_labels, holdout_patients, diagnoses)

# Evaluate holdout dataset
accuracy, cm, report = evaluate_holdout(holdout_data, optimal_threshold)

print(f"Holdout accuracy: {accuracy:.4f}")
print("Confusion Matrix (holdout):")
print(cm)
print("Classification Report (holdout):")
print(report)
