This project provides an end-to-end machine learning pipeline to analyze and predict heart disease using the UCI Heart Disease dataset. It covers data preprocessing, feature selection, dimensionality reduction (PCA), supervised learning (Logistic Regression, Decision Tree, Random Forest, SVM), unsupervised clustering (K-Means, Hierarchical), model evaluation, hyperparameter tuning, and deployment via a Streamlit web application for real-time predictions.

##Features

Data Preprocessing & Cleaning: Handle missing values, encode categorical variables, scale numerical features.

Dimensionality Reduction: Apply PCA to reduce feature dimensionality.

Feature Selection: Use RFE, Chi-Square, and feature importance methods.

Supervised Learning: Train and evaluate Logistic Regression, Decision Tree, Random Forest, and SVM models.

Unsupervised Learning: Apply K-Means and Hierarchical clustering to discover patterns.

Hyperparameter Tuning: Optimize models using GridSearchCV and RandomizedSearchCV.

Deployment: Streamlit UI for user input and live predictions, optional Ngrok deployment.

Model Export: Save trained models in .pkl format for reuse.

##File Structure
Heart_Disease_Project/
│── data/
│   └── heart_disease.csv
│── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
│── models/
│   └── final_model.pkl
│── ui/
│   └── app.py
│── deployment/
│   └── ngrok_setup.txt
│── results/
│   └── evaluation_metrics.txt
│── README.md
│── requirements.txt
│── .gitignore

##Installation

Clone the repository:

git clone https://github.com/yourusername/Heart_Disease_Project.git


##Install dependencies:

pip install -r requirements.txt


Run Jupyter notebooks for exploration or training models.

##Usage

Run Streamlit UI:

streamlit run ui/app.py


##Dataset

The project uses the UCI Heart Disease Dataset
.

