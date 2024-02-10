# Implemmentation of Deep learning framework for land classification of high-resolution aerial images using torch lightning 


## Overview

This repository is done for fine-tuning a CNN employing U-Net architecture with VGG-19 encoder to classify the aerial images of the Norwegian onshore wind farms pre and post-development and quantify their direct occupation.

## Data 

The training data was acquired from the aerial images from the Norwegian aerial image portal (Norge i Bilder). Label data were acquired from the AR5/FKB dataset by the Norwegian Institute of Bioeconomy Research (NIBIO, 2023).

## Data Preparation 
To prepare the training data for our model, we employed the following preprocessing steps using GDAL:

1. **Selection of Areas:** We selected 4 random areas in Norway where the labels in the AR5/FKB dataset match the input from aerial images obtained from Norge i Bilder. This ensures that our model trains on accurately matched data for land use classification.

2. **Image Preprocessing:** The aerial images were then preprocessed by:
   - **Clipping:** Extracting the relevant sections from the larger images that correspond to the selected areas.
   - **Cutting into Patches:** Dividing the clipped images into smaller patches of size 512x512 pixels. This step is crucial for managing the computational load and improving the efficiency of our CNN model during training.
  
3. **Data Augmentation:** To enhance the robustness of our model and increase the diversity of our training dataset, we performed data augmentation on the preprocessed patches. This was achieved by:
   - **Rotation:** Rotating the patches by 90 degrees. This step introduces variability in the dataset, simulating different orientations that the model might encounter, thus improving its generalization capabilities.

## Model Training

After preparing and augmenting our dataset, we proceeded to train our convolutional neural network model. Here's an overview of the training process:

1. **Number of Epochs:** The model was trained for 100 epochs. 

2. **Loss Function:** We used the Cross-Entropy Loss function to train the model.

## Model Inference 

## Quick Start Guide

### Step 1: Download the Model Checkpoint
Download the trained model checkpoint from our Hugging Face repository to use for predictions:
- [Hugging Face Model Repository](https://huggingface.co/Swardhunter/UNET_VGG199/tree/main)

### Step 2: Set Up Your Environment
1. **Install Required Libraries:**
   - Ensure you have Python and pip installed on your system. Then, install the required libraries listed in the `requirements.txt` file using pip:
     ```sh
     pip install -r requirements.txt
     ```

2. **Use the Prediction Notebook:**
   - Open and run the `Prediction.ipynb` notebook from the repository for easy model predictions. Make sure Jupyter Notebook is installed:
     ```sh
     pip install notebook
     ```
   - Launch Jupyter Notebook and navigate to `Prediction.ipynb` to start making predictions.
