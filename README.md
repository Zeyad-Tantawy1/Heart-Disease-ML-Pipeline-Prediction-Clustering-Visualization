# 🫀 Heart Disease Prediction Project  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview  
This project applies **Machine Learning** techniques to the UCI Heart Disease dataset to predict the likelihood of heart disease.  
It demonstrates the **end-to-end ML workflow**: from raw data ingestion → preprocessing → feature engineering → model building → evaluation → deployment with Streamlit.  

---

## 🎯 Objectives  
- Build a **predictive model** to classify patients with and without heart disease.  
- Apply **data preprocessing and cleaning** to ensure high-quality inputs.  
- Explore the data through **EDA** to uncover hidden patterns.  
- Perform **dimensionality reduction (PCA)** and multiple **feature selection** techniques.  
- Train, tune, and evaluate supervised learning models.  
- Explore **unsupervised learning** (clustering) for pattern discovery.  
- Deploy the final model in an **interactive Streamlit app**.  

---

## 📂 Project Structure  
```
Sprints/
│── data/                  # Datasets (raw & preprocessed)
│── models/                # Saved ML models and pipelines
│── src/                   # Source code modules
│   ├── data_loader.py     # Load raw dataset
│   ├── preprocessing.py   # Data cleaning & preprocessing
│   ├── eda.py             # Exploratory Data Analysis
│   ├── pca_module.py      # PCA dimensionality reduction
│   ├── feature_selection.py # RF Importance, RFE, Chi-Square
│   ├── supervised_models.py # ML training & evaluation
│   ├── tuning.py          # Hyperparameter tuning
│   ├── clustering.py      # K-Means & Hierarchical clustering
│   └── utils.py           # Helper functions
│── main.py                # Main script to run the pipeline
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
```

---

## ⚙️ Workflow  
1. **Data Loading** → Load and combine heart disease datasets.  
2. **Preprocessing** → Handle missing values, encoding, scaling.  
3. **EDA** → Statistical analysis and data visualization.  
4. **Dimensionality Reduction** → PCA for reducing complexity.  
5. **Feature Selection** → Random Forest importance, RFE, Chi-Square.  
6. **Model Training** → Logistic Regression, Random Forest, SVM.  
7. **Hyperparameter Tuning** → GridSearchCV & RandomizedSearchCV.  
8. **Clustering** → K-Means and Hierarchical clustering.  
9. **Deployment** → Interactive prediction app with Streamlit.  

---

## 📊 Results  
- Achieved high predictive accuracy using **Random Forest & SVM**.  
- PCA reduced dimensionality while retaining performance.  
- Feature selection improved interpretability and reduced noise.  
- Deployed model allows users to input patient data and get predictions in real-time.  

---

## 💻 Tech Stack  
- **Programming:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit  
- **Techniques:** PCA, Feature Selection, Clustering, Hyperparameter Tuning  
- **Deployment:** Streamlit  

---

## 🚀 How to Run  
1. Clone the repository:  
   ```bash
   git clone <repo-link>
   cd Sprints
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the pipeline:  
   ```bash
   python main.py
   ```

4. Launch the Streamlit app:  
   ```bash
   streamlit run app.py
   ```

---

## 📌 Future Improvements  
- Integrate **Deep Learning models** for comparison.  
- Enhance deployment with **Docker & cloud hosting**.  
- Improve UI for better user experience.  

---

## 👨‍💻 Author  
Developed by **Zeyad Tantawy** as part of the **Sprints AI & Machine Learning Program**.  
Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/zeyad-tantawy-6a5859314/).  

---

#machineLearning #DataScience #AI #HealthcareAI #Python #Sprints  
