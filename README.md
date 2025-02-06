# Breast Cancer Prediction Model  

## Table of Contents  
- [Description](#description) 
- [Source](#source) 
- [Dataset Details](#dataset-details)
- [Installation Guide](#installation-guide)  
- [Model and Methodologies](#model-and-methodologies)  
- [Outcomes and Evaluation](#outcomes-and-evaluation)  
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