# Breast Cancer Prediction Model  

## Table of Contents  
- [Description](#description) 
- [Source](#source) 
- [Dataset Details](#dataset-details)
- [Installation Guide](#installation-guide)  
- [Model and Methodologies](#model-and-methodologies)  
- [Outcomes and Evaluation](#outcomes-and-evaluation)
- [Improvements](#improvements)
    - [Tune KNN Model Using RandomizedSearchCV for Hyperparameter Optimization](#tune-knn-model-using-randomizedsearchcv-for-hyperparameter-optimization)
    - [Implement MICE for missing value imputation](#implement-mice-for-missing-value-imputation)
- [How to Use](#how-to-use)  
- [Future Directions](#future-directions)

## Description  
The **Breast Cancer Wisconsin (Original) Dataset** is a well-known dataset used for binary classification tasks to predict the diagnosis of breast cancer. It contains cases derived from fine needle aspirate samples, providing critical information to help differentiate between benign and malignant tumors.  

### Common Use Cases  
- Predicting whether a breast mass is benign or malignant using machine learning algorithms.  
- Identifying the most relevant features for classification.  
- Evaluating the performance of different machine learning models on a standardized dataset.

### Limitations  
- The dataset's simplicity may not reflect the complexity of real-world scenarios, limiting generalization.

## Source  
You can download the dataset from the UCI Machine Learning Repository: [Breast Cancer Wisconsin (Original) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original))  
  

## Dataset Details  
- **Sample Size**: 699 samples  
- **Features**:  
  - **ID number**: An identifier for each sample.  
  - **Clump Thickness**: A measurement of the thickness of the clump of cells.  
  - **Uniformity of Cell Size**: Measures how uniform the size of the cells is.  
  - **Uniformity of Cell Shape**: Assesses how uniform the shape of the cells is.  
  - **Marginal Adhesion**: The adhesion of the cells at the margins.  
  - **Single Epithelial Cell Size**: The size of a single epithelial cell.  
  - **Bare Nuclei**: The presence of nuclei without surrounding cytoplasm.  
  - **Bland Chromatin**: The texture of the chromatin in the cell nucleus.  
  - **Normal Nucleoli**: The presence of normal nucleoli in the cells.  
  - **Mitoses**: The count of cells undergoing mitosis.  
- **Target Variable**: Class label indicating whether the tumor is **benign (2)** or **malignant (4)**.  
- **Data Format**: CSV  

## Installation Guide  
To run this project, ensure you have Python installed along with Jupyter Notebook. You'll also need the following libraries:  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn

To install these libraries, you can use pip:  
```bash  
pip install pandas numpy scikit-learn matplotlib seaborn 
```

## Model and Methodologies 
In this analysis, the **K-Nearest Neighbors (KNN)** algorithm is used for modeling and classification purposes. Key steps include: 
- Data pre-processing: Handling missing values, detecting outliers, data normalization and feature selection to identify the most significant attributes, and splitting the data into training and testing sets.  
- Model training: Using a K-Nearest Neighbors (KNN) classifier to train the model on the training set.  
- Model evaluation: Assessing the model's performance using accuracy, precision, recall, F1-score, classification report, confusion matrix, and AUC score.  

## Outcomes and Evaluation

The model achieved an accuracy of 94% on the test set. Key performance metrics include:
- Precision: 0.95
- Recall: 0.93
- F1-score: 0.94
- AUC score: 0.94

Detailed performance metrics and visualizations are available in the results section of the repository.

## Improvements

This section aims to showcase several more robust models and data preprocessing techniques developed in this project, designed to enhance both the accuracy and interpretability of the results.

1. **Hyperparameter tuning for the K-Nearest Neighbor (KNN) algorithm.**
2. **Implement MICE for missing value imputation**

### Tune KNN Model Using RandomizedSearchCV for Hyperparameter Optimization

In this updated version of the model, I have enhanced the K-Nearest Neighbors (KNN) algorithm through hyperparameter tuning using `RandomizedSearchCV`. This tuning method allows for a more efficient exploration of hyperparameter space, leading to the identification of optimal settings that improve model accuracy and performance.

#### Hyperparameter Tuning Details

- **n_neighbors**: Tested a range from 1 to 10 to determine the most effective number of neighbors for classification.
- **weights**: Evaluated both 'uniform' and 'distance' weighting to analyze their impact on predictions.
- **metric**: Explored multiple distance metrics, including 'euclidean', 'manhattan', and 'minkowski'.
- **algorithm**: Compared various algorithms such as 'auto', 'ball_tree', 'kd_tree', and 'brute' to find the most efficient method for neighbor search.
- **p_value**: Incorporated values for the Minkowski distance power parameter, specifically 1 for Manhattan and 2 for Euclidean distances.
- **leaf_size**: Adjusted the leaf size for tree-based algorithms, examining a range from 1 to 21.

#### Results Overview

Despite the comprehensive tuning efforts, the model's performance metrics remained consistent with previous evaluations:  
- **KNN Testing Accuracy**: 94.12%  
- **KNN Tuning Accuracy**: 94.12%  
- **AUC Score**: 94.15%  
- **AUC after Tuning**: 94.15%  

These results indicate that, while tuning practices are beneficial for exploring the potential of the model, this particular process did not yield an improvement in accuracy. This emphasizes the robustness of the initial settings and highlights the ongoing challenge of optimizing model performance in machine learning. Future work may explore additional feature engineering or alternative algorithms to further enhance predictive capabilities.

### Implement MICE for missing value imputation

A recent update includes a method to handle missing values using Multiple Imputation by Chained Equations (MICE). MICE allows for a more robust handling of missing data by generating multiple imputed datasets and combining the results for improved accuracy and reliability. This approach enhances the overall model performance and provides more credible predictions, making it particularly beneficial for analyses where missing data is a concern.

## How to Use

To use this project, clone the repository using the following command in your terminal or command prompt:

1. Clone the repository:
   ```bash
   git clone https://github.com/Ehsan-Behzadi/Breast-Cancer-Prediction-Model.git  
   cd Breast-Cancer-Prediction-Model
   ```
Next, open the Jupyter Notebook file (usually with a .ipynb extension) using Jupyter Notebook.   

2. To start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Future Directions

Future improvements and directions for this project include:
- Exploring other classification algorithms such as Random Forest, Support Vector Machine (SVM), and more.
- Hyperparameter tuning to optimize model performance.
- Incorporating additional features to enhance prediction accuracy. 