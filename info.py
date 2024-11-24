import numpy as np

# Cargar los archivos .npz para Cropped
cropped_labels = np.load("Outputs/Cropped_labels_red.npz")['labels_cropped']
cropped_images = np.load("Outputs/Cropped_images_red.npz")['images_cropped']
cropped_patients_ids = np.load("Outputs/Cropped_patient_ids.npz")['patient_ids_cropped']

# Cargar los archivos .npz para Annotated
annotated_labels = np.load("Outputs/Annotated_Labels_red.npz")['labels']
annotated_images = np.load("Outputs/Annotated_Images_red.npz")['images']
annotated_patients_ids = np.load("/fhome/vlia09/MyVirtualEnv/Outputs/Annotated_patient_ids.npz")['patient_ids_annotated']

# Cargar los archivos .npz para HoldOut
holdout_labels = np.load("Outputs/HoldOut_labels_red.npz")['labels_HOLDOUT']
holdout_images = np.load("Outputs/HoldOut_images_red.npz")['images_HOLDOUT']
holdout_patients_ids = np.load("Outputs/HoldOut_patient_ids.npz")['patient_ids_holdout']

# Mostrar la cantidad total de etiquetas y datos para Cropped
print("Total de etiquetas (Cropped_labels):", len(cropped_labels))
print("Total de imagenes (Cropped_images):", len(cropped_images))
print("Total de pacientes (Cropped_patient_ids):", len(cropped_patients_ids))

# Contar la cantidad de patient_ids unicos en Cropped
unique_patients_cropped, counts_patients_cropped = np.unique(cropped_patients_ids, return_counts=True)
print("\nConteo de pacientes (Cropped):")
for patient_id, count in zip(unique_patients_cropped, counts_patients_cropped):
    print(f"Paciente {patient_id}: {count} archivos")

# Contar la cantidad de etiquetas de cada clase en Cropped
unique_labels_cropped, counts_cropped = np.unique(cropped_labels, return_counts=True)

# Mostrar el conteo de etiquetas para Cropped
print("\nConteo de etiquetas (Cropped_labels):")
for label, count in zip(unique_labels_cropped, counts_cropped):
    print(f"Etiqueta {label}: {count} archivos")

# Mostrar la cantidad total de etiquetas y datos para Annotated
print("\nTotal de etiquetas (Annotated_labels):", len(annotated_labels))
print("Total de imagenes (Annotated_images):", len(annotated_images))
print("Total de pacientes (Annotated_patient_ids):", len(annotated_patients_ids))

# Contar la cantidad de patient_ids unicos en Annotated
unique_patients_annotated, counts_patients_annotated = np.unique(annotated_patients_ids, return_counts=True)
print("\nConteo de pacientes (Annotated):")
for patient_id, count in zip(unique_patients_annotated, counts_patients_annotated):
    print(f"Paciente {patient_id}: {count} archivos")

# Contar la cantidad de etiquetas de cada clase en Annotated
unique_labels_annotated, counts_annotated = np.unique(annotated_labels, return_counts=True)

# Mostrar el conteo de etiquetas para Annotated
print("\nConteo de etiquetas (Annotated_labels):")
for label, count in zip(unique_labels_annotated, counts_annotated):
    print(f"Etiqueta {label}: {count} archivos")

# Mostrar la cantidad total de etiquetas y datos para HoldOut
print("\nTotal de etiquetas (HoldOut_labels):", len(holdout_labels))
print("Total de imagenes (HoldOut_images):", len(holdout_images))
print("Total de pacientes (HoldOut_patient_ids):", len(holdout_patients_ids))

# Contar la cantidad de patient_ids unicos en HoldOut
unique_patients_holdout, counts_patients_holdout = np.unique(holdout_patients_ids, return_counts=True)
print("\nConteo de pacientes (HoldOut):")
for patient_id, count in zip(unique_patients_holdout, counts_patients_holdout):
    print(f"Paciente {patient_id}: {count} archivos")

# Contar la cantidad de etiquetas de cada clase en HoldOut
unique_labels_holdout, counts_holdout = np.unique(holdout_labels, return_counts=True)

# Mostrar el conteo de etiquetas para HoldOut
print("\nConteo de etiquetas (HoldOut_labels):")
for label, count in zip(unique_labels_holdout, counts_holdout):
    print(f"Etiqueta {label}: {count} archivos")