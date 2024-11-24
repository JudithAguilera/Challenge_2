import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, Sampler
from sklearn.model_selection import StratifiedShuffleSplit

class StratifiedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        """
        Custom Sampler that ensures stratified sampling per batch.
        Args:
            labels (Tensor): Tensor of labels for the dataset.
            batch_size (int): The batch size.
        """
        self.labels = labels
        self.batch_size = batch_size
        
        # Get the indices for each class
        self.class_indices = {}
        for idx, label in enumerate(labels):
            if label.item() not in self.class_indices:
                self.class_indices[label.item()] = []
            self.class_indices[label.item()].append(idx)

        # Get the number of batches per class (max number of batches needed to cover all indices)
        self.num_batches = len(labels) // batch_size
        if len(labels) % batch_size != 0:
            self.num_batches += 1

    def __iter__(self):
        # Create indices for each batch, ensuring stratified sampling
        batch_indices = []
        
        # Determine the number of samples per class in each batch
        batch_class_counts = {label: 0 for label in self.class_indices}

        for _ in range(self.num_batches):
            batch = []
            
            for label, indices in self.class_indices.items():
                # Calculate how many samples from this class should be in this batch
                class_batch_size = self.batch_size // len(self.class_indices)
                # Get a random sample from the class indices
                sampled_indices = np.random.choice(indices, class_batch_size, replace=False)
                batch.extend(sampled_indices)
            
            # Shuffle the batch to avoid having all samples from the same class in one batch
            np.random.shuffle(batch)
            batch_indices.extend(batch)
        
        return iter(batch_indices)

    def __len__(self):
        return self.num_batches

# Function to load cropped data
def load_cropped_data(images_path, labels_path, batch_size=64, only_sanos=False, max_images=None, device=None):
    # Load the .npz files for Cropped
    cropped_labels = np.load(labels_path)['labels_cropped']
    cropped_images = np.load(images_path)['images_cropped']
    
    if only_sanos:
        # Filter images with label = 0 (healthy)
        sanos_indices = cropped_labels == 0
        images_sanos = cropped_images[sanos_indices]
        labels_sanos = cropped_labels[sanos_indices]
        
        # Limit to the specified maximum number of healthy images
        if max_images is not None and len(images_sanos) > max_healthy_images:
            images_sanos = images_sanos[:max_images]  # Select only the first 'max_healthy_images' images
            labels_sanos = labels_sanos[:max_images]  # Select the corresponding labels
        
        print(f"Number of healthy images: {len(images_sanos)}")
    else:
        if max_images is not None:
          images_sanos = cropped_images[:max_images]
          labels_sanos = cropped_labels[:max_images]        
        
        else: 
          images_sanos = cropped_images
          labels_sanos = cropped_labels

    # Convert the images to tensors and change the dimension order
    images_sanos_tensor = torch.tensor(images_sanos, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    
    # Convert labels to tensor
    labels_sanos_tensor = torch.tensor(labels_sanos, dtype=torch.long)

    # If the images have 1 channel, duplicate to convert them to 3 channels
    if images_sanos_tensor.shape[1] == 1:  # (N, 1, H, W)
        images_sanos_tensor = images_sanos_tensor.repeat(1, 3, 1, 1)  # (N, 3, H, W)
    
    # Move the images and labels to the appropriate device if provided
    if device:
        images_sanos_tensor = images_sanos_tensor.to(device)
        labels_sanos_tensor = labels_sanos_tensor.to(device)

    # Create and return a TensorDataset with images and corresponding labels
    dataset = TensorDataset(images_sanos_tensor, labels_sanos_tensor)
    return dataset


# Function to load Annotated data
def load_annotated_data(images_path, labels_path, device, patients_path=None, batch_size=64, max_images=None):
    # Load the .npz files for Annotated
    annotated_labels = np.load(labels_path)['labels']
    annotated_images = np.load(images_path)['images']
    
    if patients_path:
        annotated_patients_ids = np.load(patients_path)['patient_ids_annotated']
    
    # If max_images is set, limit the number of images and labels
    if max_images is not None:
        print(f"Limiting the dataset to the first {max_images} images.")
        annotated_images = annotated_images[:max_images]
        annotated_labels = annotated_labels[:max_images]

    # Convert the images to tensors and move to the appropriate device
    images_tensor = torch.tensor(annotated_images).float().to(device)
    images_tensor = images_tensor.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
    labels_tensor = torch.tensor(annotated_labels).long().to(device)

    # Create a dataset and dataloader for batches
    dataset = TensorDataset(images_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# Funcion para cargar los datos de HoldOut
def load_holdout_data(images_path, labels_path, patients_path, device, batch_size = 64, max_images=None, stratified=False):
    # Cargar los archivos .npz para HoldOut
    holdout_labels = np.load(labels_path)['labels_HOLDOUT']
    holdout_images = np.load(images_path)['images_HOLDOUT']
    holdout_patients_ids = np.load(patients_path)['patient_ids_holdout']

    
    # If max_images is set, limit the number of images and labels
    if stratified is not None:
        # Apply stratified sampling first to ensure class balance
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=None, train_size=None, random_state=42)
        
        # Stratified shuffle split to ensure balance of classes
        indices = list(stratified_split.split(holdout_images, holdout_labels))[0][0]
        stratified_images = holdout_images[indices]
        stratified_labels = holdout_labels[indices]
        if max_images is not None:
            print(f"Limiting the dataset to the first {max_images} images.")
            stratified_images = stratified_images[:max_images]
            stratified_labels = stratified_labels[:max_images]
        # Convert images to tensor and move to the correct device (CPU/GPU)
        images_tensor = torch.tensor(stratified_images).float().to(device)
        images_tensor = images_tensor.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
        labels_tensor = torch.tensor(stratified_labels).long().to(device)
        
    else:
        if max_images is not None:
            print(f"Limiting the dataset to the first {max_images} images.")
            holdout_images = holdout_images[:max_images]
            holdout_labels = holdout_labels[:max_images]
        # Convert images to tensor and move to the correct device (CPU/GPU)
        images_tensor = torch.tensor(holdout_images).float().to(device)
        images_tensor = images_tensor.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
        labels_tensor = torch.tensor(holdout_labels).long().to(device)


    # Crear un dataset y dataloader para batches
    dataset = TensorDataset(images_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader
