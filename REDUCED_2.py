import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import re 
from sklearn.utils import shuffle  





def clean_folder_name(folder_name):
    # Remove numeric suffixes at the end of the folder name
    return re.sub(r'_\d+$', '', folder_name)

def output_folder(path):
    # Check if the output path exists. if not, create it
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Output directory created: {path}")
    else:
        print(f"Output directory already exists: {path}")
    return

def clean_image_name(image_name):
    # Remove the file extension
    image_name = os.path.splitext(image_name)[0]
    # Remove leading zeros
    image_name = image_name.lstrip('0')
    return image_name

def process_cropped(diagnosis_path, cropped_path, output_folder, max_images=3000):
    print(f"Starting to process cropped images from {cropped_path}...")
    
    try:
        density_df = pd.read_csv(diagnosis_path)
        density_df.columns = density_df.columns.str.strip()
        print("CSV file loaded successfully.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        
    density_map = {row['CODI']: 1 if row['DENSITAT'] == 'ALTA' else 0 for _, row in density_df.iterrows()}
    print("Density map loaded:", density_map)

    patient_folders = sorted(os.listdir(cropped_path))
    quarter_index = len(patient_folders) // 4
    selected_folders = patient_folders[:quarter_index]

    images_cropped = []
    labels_cropped = []
    patient_ids_cropped = []
    total_images = 0

    for patient_folder in selected_folders:
        patient_path = os.path.join(cropped_path, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        patient_id = patient_folder.split('_')[0]
        label = density_map.get(patient_id, 0)
        image_count = 0

        for image_file in os.listdir(patient_path):
            if image_file.endswith('.png'):
                image_path = os.path.join(patient_path, image_file)

                with Image.open(image_path) as img:
                    img = img.convert('RGB')
                    img = img.resize((224, 224))
                    img_array = np.array(img) / 255.0

                    images_cropped.append(img_array)
                    labels_cropped.append(label)
                    patient_ids_cropped.append(patient_id)
                    image_count += 1
                    total_images += 1

    images_cropped = np.array(images_cropped)
    labels_cropped = np.array(labels_cropped)

    images_path = os.path.join(output_folder, "Cropped_images_red.npz")
    labels_path = os.path.join(output_folder, "Cropped_labels_red.npz")
    patient_ids_path = os.path.join(output_folder, "Cropped_patient_ids.npz")

    np.savez_compressed(images_path, images_cropped=images_cropped)
    np.savez_compressed(labels_path, labels_cropped=labels_cropped)
    np.savez_compressed(patient_ids_path, patient_ids_cropped=patient_ids_cropped)

    print(f"Total cropped images processed: {len(images_cropped)}")
    print(f"Cropped image files saved successfully.")


def process_annotated(patch_path, annotated_folder, output_folder, max_images=1000):
    print(f"Starting to process annotated images from {annotated_folder}...")
    
    try:
        patch_data = pd.read_excel(patch_path, engine='openpyxl')
        print("Excel file loaded successfully.")
    except Exception as e:
        print(f"Error loading the Excel file: {e}")

    columns_of_interest = ['Pat_ID', 'Window_ID', 'Presence']
    patch_data_filtered = patch_data[columns_of_interest]
    grouped_data = patch_data_filtered.groupby('Pat_ID')

    folder_list = sorted(os.listdir(annotated_folder))
    selected_folders = folder_list  # Usar todas las carpetas en lugar de un tercio

    images, labels = [], []
    patient_ids_annotated = []
    image_count = 0
    label_counts = {0: 0, 1: 0}  # Para contar etiquetas 0 y 1
    patient_counts = {}  # Para contar imagenes por paciente

    for folder_name in selected_folders:
        folder_path = os.path.join(annotated_folder, folder_name)
        
        if os.path.isdir(folder_path):
            cleaned_folder_name = clean_folder_name(folder_name)
            if cleaned_folder_name in grouped_data.groups:
                print(f"Processing folder: {folder_name}")
                
                pat_data = grouped_data.get_group(cleaned_folder_name)
                window_ids = pat_data['Window_ID'].tolist()
                presence_labels = pat_data['Presence'].tolist()
                window_ids = [str(win_id) for win_id in window_ids]

                for image_file in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_file)
                    
                    if image_file.endswith(('.png', '.jpg', '.jpeg')):
                        cleaned_image_name = clean_image_name(image_file)
                        if cleaned_image_name in window_ids:
                            window_id_index = window_ids.index(cleaned_image_name)

                            label = 0 if presence_labels[window_id_index] == -1 else 1
                            images.append(cv2.resize(cv2.imread(image_path), (224, 224)))
                            labels.append(label)
                            patient_ids_annotated.append(cleaned_folder_name)
                            image_count += 1
                            
                            # Actualizar el recuento de etiquetas
                            label_counts[label] += 1

                            # Actualizar el recuento de imagenes por paciente
                            if cleaned_folder_name in patient_counts:
                                patient_counts[cleaned_folder_name] += 1
                            else:
                                patient_counts[cleaned_folder_name] = 1

    images = np.array(images)
    labels = np.array(labels)

    images, labels = shuffle(images, labels, random_state=42)

    images_path = os.path.join(output_folder, "Annotated_Images_red.npz")
    labels_path = os.path.join(output_folder, "Annotated_Labels_red.npz")
    patient_ids_path = os.path.join(output_folder, "Annotated_patient_ids.npz")

    np.savez_compressed(images_path, images=images)
    np.savez_compressed(labels_path, labels=labels)
    np.savez_compressed(patient_ids_path, patient_ids_annotated=patient_ids_annotated)

    print(f"Total annotated images: {len(images)}")
    print(f"Annotated image files saved successfully.")
    
    # Imprimir los recuentos de etiquetas
    print(f"Label 0 count: {label_counts[0]}")
    print(f"Label 1 count: {label_counts[1]}")

    # Imprimir el recuento de imagenes por paciente
    print("Image count per patient:")
    for patient_id, count in patient_counts.items():
        print(f"  {patient_id}: {count} images")



def process_holdOut(csv_HOLDOUT, HOLDOUT_path, output_folder, max_images=1000):
    print(f"Starting to process holdout images from {HOLDOUT_path}...")
    
    density_df = pd.read_csv(csv_HOLDOUT)
    density_df.columns = density_df.columns.str.strip()

    density_map = {row['CODI']: 1 if row['DENSITAT'] == 'ALTA' else 0 for _, row in density_df.iterrows()}
    print("Density map loaded:", density_map)

    patient_folders = sorted(os.listdir(HOLDOUT_path))
    quarter_index = len(patient_folders) // 4
    selected_folders = patient_folders[:quarter_index]

    images_HOLDOUT = []
    labels_HOLDOUT = []
    patient_ids_holdout = []

    for patient_folder in selected_folders:
        patient_path = os.path.join(HOLDOUT_path, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        patient_id = patient_folder.split('_')[0]
        label = density_map.get(patient_id, 0)

        for image_file in os.listdir(patient_path):
            if image_file.endswith('.png'):
                image_path = os.path.join(patient_path, image_file)

                with Image.open(image_path) as img:
                    img = img.convert('RGB')
                    img = img.resize((224, 224))
                    img_array = np.array(img) / 255.0

                    images_HOLDOUT.append(img_array)
                    labels_HOLDOUT.append(label)
                    patient_ids_holdout.append(patient_id)

    images_HOLDOUT = np.array(images_HOLDOUT)
    labels_HOLDOUT = np.array(labels_HOLDOUT)

    images_path = os.path.join(output_folder, "HoldOut_images_red.npz")
    labels_path = os.path.join(output_folder, "HoldOut_labels_red.npz")
    patient_ids_path = os.path.join(output_folder, "HoldOut_patient_ids.npz")

    np.savez_compressed(images_path, images_HOLDOUT=images_HOLDOUT)
    np.savez_compressed(labels_path, labels_HOLDOUT=labels_HOLDOUT)
    np.savez_compressed(patient_ids_path, patient_ids_holdout=patient_ids_holdout)

    print(f"Total holdout images processed: {len(images_HOLDOUT)}")
    print(f"Holdout image files saved successfully.")


# Paths
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

cropped_path = "/fhome/vlia/HelicoDataSet/CrossValidation/Cropped/"
annotated_path = "/fhome/vlia/HelicoDataSet/CrossValidation/Annotated/"
HOLDOUT_path = "/fhome/vlia/HelicoDataSet/HoldOut/"
patch_data_path = "HP_WSI-CoordAllAnnotatedPatches.xlsx"
diagnosis_data_path = "/fhome/vlia/HelicoDataSet/PatientDiagnosis.csv"
output_folder_path = "/fhome/vlia09/MyVirtualEnv/Outputs"

output_folder(output_folder_path)

#process_cropped(diagnosis_data_path, cropped_path, output_folder_path)
process_annotated(patch_data_path, annotated_path, output_folder_path)
#process_holdOut(diagnosis_data_path, HOLDOUT_path, output_folder_path)
