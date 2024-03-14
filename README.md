# Sentiment Analysis of IMDB Reviews

## Introduction
This project focuses on sentiment analysis using three different machine learning models: Logistic Regression, Support Vector Machine (SVM), and Long Short-Term Memory (LSTM). The aim is to classify movie reviews from the IMDB dataset into positive or negative sentiments.

## Dataset
The dataset used is the 'aclImdb' dataset, containing a large number of IMDB reviews. This dataset is split into training and testing subsets for model evaluation.

## Project Structure
The project consists of several key steps:

- **Data Loading:** Loading the IMDB dataset and creating training and testing datasets.
- **Data Preprocessing:** Preprocessing steps such as converting to lowercase, removing non-alphabet characters, tokenization, removing stop words, and lemmatization.
- **Feature Engineering:** Vectorization of preprocessed text data using TF-IDF (Term Frequency-Inverse Document Frequency).
- **Model Training and Evaluation:**
  - Logistic Regression Model: Implementation and evaluation.
  - Support Vector Machine (SVM) Model: Implementation and evaluation.
  - Long Short-Term Memory (LSTM) Model: Implementation using Keras and TensorFlow, followed by evaluation.

## Results and Observations
The notebook contains detailed observations and comments on the performance of each model. It includes analyses of training loss, training accuracy, test loss, and test accuracy. A comparative discussion on overfitting in models and their respective accuracies on training and test data is provided. Insights on the suitability and limitations of each model for sentiment analysis tasks are discussed.

## Conclusion
The project concludes with recommendations on model selection based on factors such as interpretability, performance, suitability for complex tasks, and computational efficiency.
