# Challenge_2
This repository contains the implementation of an autoencoder model for classifying images into healthy and pathological categories. The project includes various scripts for training, evaluating, and analyzing the performance of the model. Below is an overview of the structure of the repository and the role of each script.

## Repository Structure
### Scripts Overview
1. final_autoencoder.py

Contains the structure of the autoencoder model used for image classification. This script defines the layers and architecture of the autoencoder.

2. final_autoencoder_training.py

Handles the training of the autoencoder model. It performs k-fold cross-validation, trains the model, and returns the final trained model along with its metrics. This script also generates various plots, such as MSE histograms, to visualize the training process and evaluate model performance.

3. final_check_autoencoder.py

Evaluates the performance of the trained autoencoder. It calculates the optimal threshold for classifying images as healthy or pathological, and returns important metrics such as:
Threshold, MSE Histogram, Confusion Matrices, ROC Curves, Accuracy Scores

4. final_dataloader.py

Contains functions for generating the necessary data loaders. This script facilitates the loading of the datasets and batching for training and evaluation.

5. final_datasets.py

Handles the preprocessing of the three primary datasets: cropped, annotated, and holdout. It generates .npz files containing the processed data, ready for use by the model.

6. final_patient_diagnosis.py

Classifies whether a patient’s diagnosis is healthy or ill. This script is similar to final_check_autoencoder.py and uses the same evaluation metrics. It also calculates the optimal threshold using the ROC curve and returns the same performance metrics.

7. info.py and plots.py

Used for generating plots and extracting information about the datasets and generated files. These scripts helps visualize and analyze the structure of the datasets, model outputs, and other useful statistics.


### Datasets
The project uses three main datasets for training and evaluation:

1. Cropped Dataset
This dataset contains images where the area of interest has been cropped out. It includes both healthy and pathological images.

2. Annotated Dataset
This dataset includes images with annotations that indicate whether the image is healthy or shows pathology.

3. Hold-Out Dataset
A separate dataset used exclusively for testing the model's generalization and performance.

These datasets are processed and stored as .npz files for easy access. During experimentation, some adjustments were made to the datasets in order to improve the model's performance, so there may be inconsistencies in the number of samples across the datasets. 


## How to Use
### Preprocessing Datasets
Run final_datasets.py to preprocess and generate the .npz files for the cropped, annotated, and holdout datasets.

### Train the Autoencoder
Run final_autoencoder_training.py to train the autoencoder on the preprocessed data. This script will generate k-fold models, the final trained model, and output various metrics and plots for analysis, including MSE histograms.

### Evaluate the Model
Use final_check_autoencoder.py to evaluate the performance of the trained autoencoder. It will provide the optimal threshold for classifying images as healthy or pathological, and output performance metrics such as confusion matrices, ROC curves, and accuracy scores.

### Patient Diagnosis
Run final_patient_diagnosis.py to classify a patient's diagnosis as healthy or ill. It will return similar performance metrics as final_check_autoencoder.py and use the ROC curve to determine the threshold.

Generate Plots and Extract Info
Use info.py and plots.py to generate plots and extract useful information about the datasets, model performance, and the files you're working with. This script helps visualize the structure of the datasets and outputs generated during training and evaluation.
