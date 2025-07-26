# Iris Flower Classification using ZenML

This project implements an end-to-end machine learning pipeline to classify iris flower species using ZenML, a framework for reproducible ML pipelines.

## Project Overview

The Iris dataset is a classic dataset in machine learning, containing 150 samples of iris flowers with features such as sepal length, sepal width, petal length, and petal width. The goal is to classify each flower into one of three species: *Iris-setosa*, *Iris-versicolor*, or *Iris-virginica*.

This project demonstrates how to use ZenML to structure and run ML workflows efficiently.


## Project Structure

Iris_Flower_Classification/
│
├── steps/
│ ├── data_loader.py # Loads data from CSV
│ ├── data_prep.py # Preprocesses and splits the data
│ ├── data_training.py # Trains RandomForest model
│ ├── predict_model.py # Generates predictions
│ └── evaluate_model.py # Evaluates model performance
│
├── pipeline.py # Pipeline definition
├── main.py # Entry point to run the pipeline
├── README.md # Project documentation
└── pyproject.toml # Project dependencies

## Features

- Uses ZenML pipelines with reusable steps
- Preprocessing with train/test split
- Trains a `RandomForestClassifier`
- Predicts on test set
- Evaluates using F1 Score, Precision, Accuracy
- Modular and scalable structure

## Model Evaluation Output

After running the pipeline, you will see printed metrics such as:

Accuracy- 1.00

F1 Score- 1.00

Precision- 1.00

Recall- 1.00

This shows a balanced model between train and test. This implies the model does not overfit.

## Dataset

The Iris dataset is publicly available and included in most ML libraries (like scikit-learn).

📌 Author

Ayokunle Adeleye
Github.com/Aycares


