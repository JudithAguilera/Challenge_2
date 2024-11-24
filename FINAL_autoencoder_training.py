import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from FINAL_dataloader import load_cropped_data, load_annotated_data, load_holdout_data
from FINAL_autoencoder import Autoencoder

# Train the autoencoder and perform evaluation for one fold
def train_autoencoder(train_loader, val_loader, fold, n_epochs=30, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = Autoencoder().to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    min_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        running_train_loss = 0.0
        for data in train_loader:
            # Get the inputs and move to device
            inputs, _ = data
            inputs = inputs.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, inputs)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()  # Liberar la memoria GPU

            running_train_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, _ = data
                inputs = inputs.to(device)

                # Forward pass
                outputs = model(inputs)

                # Calculate loss
                loss = criterion(outputs, inputs)
                running_val_loss += loss.item()

        # Calculate average validation loss for the epoch
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Check for early stopping
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"autoencoder_fold_{fold}_best.pth")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} for fold {fold}")
            break

        print(f"Epoch {epoch + 1}/{n_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save loss plot for the current fold
    print(f"Saving loss plot for fold {fold}...")
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.title(f"Loss Plot - Fold {fold}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"loss_plot_fold_{fold}.png")
    plt.close()
    
    
    # Calculate reconstruction errors for this fold
    all_reconstruction_errors = []
    model.eval()
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            reconstruction_errors = torch.mean((outputs - images) ** 2, dim=[1, 2, 3]).cpu().numpy()
            all_reconstruction_errors.extend(reconstruction_errors)

    # Calculate metrics of the errors
    mse_mean = np.mean(all_reconstruction_errors)
    mse_std = np.std(all_reconstruction_errors)
    print(f"Fold {fold}: MSE Mean = {mse_mean:.4f}, MSE STD = {mse_std:.4f}")

    # Plot reconstruction error distribution for this fold
    plt.figure(figsize=(8, 6))
    plt.hist(all_reconstruction_errors, bins=50, alpha=0.7, color='blue', label='Reconstruction Errors')
    plt.axvline(mse_mean, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mse_mean:.4f}')
    plt.axvline(mse_mean - mse_std, color='green', linestyle='dashed', linewidth=1.5, label=f'-1 STD: {mse_mean - mse_std:.4f}')
    plt.axvline(mse_mean + mse_std, color='green', linestyle='dashed', linewidth=1.5, label=f'+1 STD: {mse_mean + mse_std:.4f}')
    plt.title(f'Reconstruction Error Distribution (Fold {fold})')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()

    # Add text with the mean and std
    plt.text(0.95, 0.95, f'Mean: {mse_mean:.4f}\nSTD: {mse_std:.4f}', 
            transform=plt.gca().transAxes, 
            fontsize=10, color='black', 
            verticalalignment='top', 
            horizontalalignment='right')

    # Save the plot for this fold
    plt.savefig(f"reconstruction_error_distribution_fold_{fold}.png")
    plt.close()

    # Return the final model and the reconstruction errors
    return model, train_losses, val_losses, all_reconstruction_errors, mse_mean, mse_std

'''
# Function to plot reconstruction error distribution
def plot_reconstruction_error_distribution(errors, mse_mean, mse_std, fold):
    print(f"Plotting reconstruction error distribution for fold {fold + 1}...")
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=50, alpha=0.7, color='blue', label='Reconstruction Errors')
    plt.axvline(mse_mean, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mse_mean:.4f}')
    plt.axvline(mse_mean - mse_std, color='green', linestyle='dashed', linewidth=1.5, label=f'-1 STD: {mse_mean - mse_std:.4f}')
    plt.axvline(mse_mean + mse_std, color='green', linestyle='dashed', linewidth=1.5, label=f'+1 STD: {mse_mean + mse_std:.4f}')
    plt.title(f'Reconstruction Error Distribution (Fold {fold + 1})')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reconstruction_error_distribution_fold_{fold + 1}.png")
    plt.close()'''


# Train the final autoencoder on the entire dataset
def train_final_autoencoder(train_loader, n_epochs=30, batch_size=64, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = Autoencoder().to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_losses = []

    for epoch in range(n_epochs):
        model.train()
        running_train_loss = 0.0
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()  # Liberar la memoria GPU


            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {avg_train_loss:.4f}")

    # Save the final loss plot
    print("Saving final loss plot...")
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.title("Final Loss Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("final_loss_plot.png")
    plt.close()

    # Save the model
    print("Saving the final trained model...")
    torch.save(model.state_dict(), "final_trained_autoencoder.pth")

    # Visualization of reconstructed images
    print("Reconstructed vs original")
    num_images = 5
    sample_images = next(iter(train_loader))[0][:num_images].to(device)      # Sample images from the training loader
    reconstructed_images = model(sample_images)
    
    # Plotting original vs reconstructed images
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        # Original images
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(sample_images[i].cpu().permute(1, 2, 0).numpy())
        plt.title("ORIGINAL")
        plt.axis("off")
    
        # Reconstructed images
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed_images[i].cpu().permute(1, 2, 0).numpy())
        plt.title("RECONSTRUCTED")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("original_vs_reconstructed_final.png")
    plt.close()


    return


# Plot training loss graph
def plot_loss_graph(train_losses):
    print("Plotting final training loss graph...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss (per batch)")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    plt.savefig("final_loss_graph.png")
    plt.close()

'''
# Calculate reconstruction errors on the entire dataset
def calculate_reconstruction_errors(train_loader, model, device):
    all_reconstruction_errors = []
    model.eval()
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            outputs = model(images)
            reconstruction_errors = torch.mean((outputs - images) ** 2, dim=[1, 2, 3]).cpu().numpy()
            all_reconstruction_errors.extend(reconstruction_errors)
    return all_reconstruction_errors

def train_final_autoencoder(train_loader, n_epochs=30, batch_size=64, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Variables para el entrenamiento
    train_losses = []

    # Early stopping
    best_train_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = model.state_dict()

    # Entrenamiento
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (images, _) in enumerate(train_loader):
            # Asegurar que las imagenes tengan el formato correcto
            if len(images.shape) == 3:
                images = images.unsqueeze(1)  # Agregar un canal si falta
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            # Sumar la perdida por lotes
            train_loss += loss.item()

            # Imprimir perdida por cada lote
            train_losses.append(loss.item())
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Train Loss = {loss.item():.4f}")

        # Calcular la perdida promedio para la epoca
        epoch_train_loss = train_loss / len(train_loader)
        print(f"EPOCH [{epoch + 1}/{n_epochs}], Epoch Train Loss = {epoch_train_loss:.4f}")

        # Early stopping logic
        if epoch_train_loss < best_train_loss:
            best_train_loss = epoch_train_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()  # Guardar los pesos del mejor modelo
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f"Early stopping: No improvement in train loss for {patience} epochs")
            break

        # Limpiar cache de CUDA
        torch.cuda.empty_cache()

    # Restaurar los mejores pesos encontrados
    model.load_state_dict(best_model_wts)

    # Guardar el modelo entrenado con todos los datos
    torch.save(model.state_dict(), "final_trained_autoencoder.pth")

    
    # Graficar la loss durante el entrenamiento
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss (per batch)")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    plt.savefig("final_loss_graph.png")
    plt.close()
    
    return'''
    
# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    cropped_images_path = 'Outputs/Cropped_images_red.npz'
    cropped_labels_path = 'Outputs/Cropped_labels_red.npz'
    
    # Cargar cropped dataset
    print("Loading cropped dataset...")
    cropped_dataset = load_cropped_data(cropped_images_path, cropped_labels_path, only_sanos=True, max_healthy_images = 7000)
    print("Cropped dataset loaded.")

    k = 6  # Number of folds
    all_mse_means = []
    all_mse_stds = []
    all_errors = []


    # Load data
    print("Setting up K-fold cross-validation...")
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    fold = 0
    for train_idx, val_idx in kfold.split(cropped_dataset):
        fold += 1

        print(f"Training fold {fold}...")

        # Create data loaders for this fold
        train_subset = Subset(cropped_dataset, train_idx)
        val_subset = Subset(cropped_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

        # Train the model on this fold
        model, train_losses, val_losses, all_reconstruction_errors, mse_mean, mse_std = train_autoencoder(train_loader, val_loader, fold)

        # Calculate MSE for this fold
        all_errors.append(all_reconstruction_errors)
        all_mse_means.append(mse_mean)
        all_mse_stds.append(mse_std)

    # Plot the combined MSE error distribution
    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap('tab10')  # Define color map
    
    # Initialize a string for fold statistics
    stats_text = "Fold Statistics:\n"
    
    for i, errors in enumerate(all_errors):
        # Plot histogram for each fold's errors
        plt.hist(errors, bins=50, alpha=0.5, label=f'Fold {i + 1}', color=colors(i / len(all_errors)))
        # Vertical line for the mean
        plt.axvline(all_mse_means[i], color=colors(i / len(all_errors)), linestyle='dashed', linewidth=1.5)
    
        # Add mean and std dev to statistics text
        stats_text += f"Fold {i + 1}: Mean = {all_mse_means[i]:.4f}, Std Dev = {all_mse_stds[i]:.4f}\n"
    
    # Title and labels
    plt.title('Combined Reconstruction Error Distributions')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Display statistics in a text box
    plt.gcf().text(0.02, 0.5, stats_text, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the combined plot
    plt.tight_layout()
    plt.savefig("combined_reconstruction_error_distributions_with_stats.png")
    plt.close()

    print("K-fold cross-validation completed.")
    
    print("Starting final training on the entire dataset...")
    # Train the final autoencoder on the entire dataset
    train_final_autoencoder(DataLoader(cropped_dataset, batch_size=64, shuffle=True), n_epochs=30, batch_size=64, patience=5)
    print("Enddd")

    



