# 📊 Predict Customer Churn - Kaggle Playground S6E3

## 📝 Overview
This repository contains my solution for the **Kaggle Playground Series - Season 6, Episode 3: Predict Customer Churn**. 
The objective of this project is to build a robust machine learning model to predict the probability of a customer leaving a service (churning) based on various customer attributes.

## 💡 Approach & Methodology
To achieve high predictive performance and prevent overfitting, I implemented a comprehensive pipeline:

1. **Data Preprocessing & Cleaning:**
   * Handled missing values using Mean Imputation for numerical features.
   * Converted categorical variables into numerical formats using `LabelEncoder`.
   * Dropped non-predictive columns (`id`, `CustomerId`, `Surname`).

2. **Advanced Feature Selection (PSO):**
   * Implemented **Particle Swarm Optimization (PSO)** using the `pyswarms` library.
   * Used PSO to search the feature space and select the optimal subset of features that maximizes the ROC AUC score, effectively reducing noise and dimensionality.

3. **Ensemble Modeling:**
   * Built a powerful ensemble model using a `VotingClassifier` (Soft Voting).
   * Combined three state-of-the-art Gradient Boosting algorithms:
     * **LightGBM:** For fast, efficient, and scalable training.
     * **XGBoost:** For deep learning capabilities on tabular data.
     * **CatBoost:** For robust handling of categorical patterns and preventing overfitting.

4. **Model Evaluation:**
   * Evaluated the model using the **ROC AUC** metric (Area Under the Receiver Operating Characteristic Curve).
   * Visualized model performance using Confusion Matrices, ROC Curves, and Feature Importance charts.

## 🛠️ Technologies & Libraries Used
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
* **Optimization:** `pyswarms` (Particle Swarm Optimization)
* **Visualization:** `matplotlib`, `seaborn`

## 🚀 How to View the Code
You can view the full workflow and code directly in the Jupyter Notebook provided in this repository:
[`Kaggle_Customer_Churn.ipynb`](./Kaggle_Customer_Churn.ipynb)

---
*Feel free to reach out or open an issue if you have any questions or suggestions for improvement!*
