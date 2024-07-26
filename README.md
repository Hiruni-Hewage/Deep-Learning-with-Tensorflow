# Chest X-Ray COVID-19 Detection

This project aims to detect COVID-19 from chest X-ray images using transfer learning and deep learning techniques. The implementation utilizes TensorFlow and various data science libraries to achieve accurate results.

## Project Overview

The goal of this project is to classify chest X-ray images into COVID-19 positive and negative categories. The project leverages pre-trained models for transfer learning to improve the accuracy and efficiency of the classification process.

## Features

- **Transfer Learning**: Utilizes pre-trained models for efficient learning.
- **Deep Learning**: Implements deep neural networks for image classification.
- **Data Visualization**: Visualizes data distribution and sample images.
- **Model Evaluation**: Provides detailed evaluation metrics for model performance.

## Dataset

The dataset used in this project is sourced from Kaggle, consisting of labeled chest X-ray images for COVID-19 and normal cases.

## Requirements

- TensorFlow
- Pandas
- Matplotlib
- Numpy
- OpenCV
- Scikit-learn

## Usage

1. **Prepare the dataset**: Download and extract the dataset from Kaggle.
2. **Run the notebook**: Open the Jupyter notebook and run the cells to execute the code.

## Code Overview

The code is organized into several sections:

1. **Imports and Setup**: Import necessary libraries and configure settings.
    ```python
    import os
    import pathlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import random
    import cv2
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    ```

2. **Data Loading and Preprocessing**: Load and preprocess the dataset.
    ```python
    # Load and preprocess data
    data_dir = pathlib.Path("/path/to/dataset")
    class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
    ```

3. **Model Building**: Build and compile the deep learning model.
    ```python
    base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    ```

4. **Model Training and Evaluation**: Train the model and evaluate its performance.
    ```python
    history = model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[EarlyStopping(patience=3)])
    ```

## Results

The model achieved significant accuracy in classifying COVID-19 from chest X-ray images. Detailed evaluation metrics and visualizations are provided in the notebook.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to Kaggle for providing the dataset.
- Inspired by various open-source deep learning projects.


