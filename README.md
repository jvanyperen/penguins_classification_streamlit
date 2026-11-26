# Penguin Species Prediction with Decision Trees

An interactive Streamlit application for exploring decision tree classification using the Palmer Penguins dataset.

The app allows users to experiment with feature selection, tune decision tree hyperparameters, evaluate model performance, visualise the resulting tree, and make predictions for new penguins.

---

## Features

### Dataset Exploration

- Loads the Palmer Penguins dataset from Seaborn
- Handles missing values automatically
- Optional display of the dataset and summary statistics

### Feature Selection

Choose which numerical features to use for model training:

- Bill Length (mm)
- Bill Depth (mm)
- Flipper Length (mm)
- Body Mass (g)

Students can investigate how feature selection impacts classification performance.

### Hyperparameter Tuning

Adjust key Decision Tree hyperparameters:

- Maximum tree depth
- Minimum samples required to split a node
- Splitting criterion (`gini` or `entropy`)

### Model Evaluation

The application automatically trains a Decision Tree classifier and displays:

- Accuracy
- Precision
- Recall
- F1 Score

### Visualisations

Interactive visualisations include:

- Confusion matrix
- Feature importance plot
- Fully rendered decision tree

These help users understand both model performance and interpretability.

### Prediction Tool

Enter measurements for a new penguin and receive:

- Predicted species
- Class probabilities for all species

---

## Learning Objectives

This application demonstrates several core machine learning concepts:

- Supervised learning
- Classification
- Decision trees
- Feature selection
- Hyperparameter tuning
- Feature importance
- Model evaluation metrics
- Probabilistic predictions

The project is intended as an educational example for undergraduate students studying data science, machine learning, statistics, or applied mathematics.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jvanyperen/penguins_classification_streamlit.git
cd penguins_classification_streamlit