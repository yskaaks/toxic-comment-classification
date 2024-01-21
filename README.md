# Toxic Comment Classification Project

## Overview

The Toxic Comment Classification project aims to create a model that can classify comments into various toxicity categories such as toxic, severe toxic, obscene, threat, insult, and identity hate. This model is valuable for online platforms seeking to maintain healthy and respectful community interactions.

## Dataset

The dataset consists of comments labeled with binary values across six categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. Each comment can belong to multiple categories or none.

## Methodology

1. **Data Exploration and Analysis**: 
   - Examining the distribution of toxic and non-toxic comments in the dataset.
   - Visualizing the data using seaborn count plots to understand the imbalance in the dataset.

2. **Preprocessing**: 
   - The data is split into training and validation sets.
   - The `TextVectorization` layer from TensorFlow is used to vectorize the comments, transforming them into token sequences.

3. **Model Building**:
   - A Sequential model is constructed using TensorFlow and Keras.
   - The model includes an Embedding layer, followed by Dropout, GlobalAveragePooling1D, Dense, and Output layers.
   - The output layer has 6 units, corresponding to the 6 categories, with a sigmoid activation function.

4. **Model Compilation**:
   - The model is compiled using the Adam optimizer and binary cross-entropy loss function, suitable for binary classification tasks.

5. **Model Training**:
   - Training is performed on the processed dataset for multiple epochs.

## Model Architecture

The model comprises the following layers:
1. `Embedding Layer`: Transforms the vectorized text data into dense vectors of fixed size.
2. `Dropout Layer`: Regularization technique to prevent overfitting.
3. `GlobalAveragePooling1D`: Reduces the dimensionality of the data.
4. `Dense Layer`: Fully connected layer with ReLU activation.
5. `Dropout Layer`: Additional dropout for regularization.
6. `Output Layer`: Final layer with 6 units (one for each category) using the sigmoid activation function to output probabilities.

## Evaluation and Metrics

- The model's performance is evaluated based on its accuracy in classifying the comments into the correct categories.
- Due to the nature of the dataset (multi-label classification), each comment can be classified into more than one category.

## Future Work

- Experimenting with different model architectures and hyperparameters to improve accuracy.
- Implementing techniques to handle class imbalance in the dataset.
- Evaluating the model with additional metrics like Precision, Recall, and F1-Score, which are more informative for imbalanced datasets.

