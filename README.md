```
# Fruit Detection Project

This project implements a Convolutional Neural Network (CNN) for the classification of plant images into different categories. It includes code for training the model, making predictions on new images, and evaluating the model's performance.

## Overview

The project consists of the following components:

1. **Training Script**: `train_model.py` - This script trains a CNN model using TensorFlow and Keras on a dataset of plant images. The trained model is saved for future use.

2. **Prediction Script**: `predict_plant.py` - This script loads the trained model and makes predictions on new images. It takes an input image and outputs the predicted plant class along with the confidence score.

3. **Pre-Trained Model**: `plant_detection_model_2.h5` - This file contains a pre-trained CNN model for plant classification. It can be used directly for making predictions without training.

4. **Data**: The project includes a dataset of plant images (`split_ttv_dataset_type_of_plants`) divided into training, testing, and validation sets.

## Usage

To use this project, follow these steps:

1. **Clone the Repository**: Clone this GitHub repository to your local machine using the following command:

   ```
   git clone https://github.com/01chaitanya01/Fruit-Detection.git
   ```

2. **Install Dependencies**: Install the required dependencies by running the following command:

   ```
   pip install -r requirements.txt
   ```

## Acknowledgments

- The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/your-kaggle-username/dataset-name).
- Inspiration and guidance for building the CNN model were derived from various online resources and tutorials.

```
