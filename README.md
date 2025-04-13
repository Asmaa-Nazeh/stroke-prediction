# 🧠 Stroke Prediction Using Machine Learning

This project aims to predict whether a person is likely to have a stroke based on various health and demographic features. It is a binary classification task using several machine learning models to identify the best-performing one.

## 📊 Dataset

The dataset used includes features such as:

- gender
- age
- hypertension
- heart_disease
- ever_married
- work_type
- Residence_type
- avg_glucose_level
- bmi
- smoking_status
- stroke (Target Variable)

📌 *Source*: [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

---

## ❓ Questions & Data Analysis

During Exploratory Data Analysis (EDA), I explored the following questions:

1. *Is the data imbalanced in terms of stroke cases?*
   - ✅ Yes, the data is highly imbalanced, and this was visualized using a bar plot.
   - 🔧 Solved using *SMOTE* oversampling technique.

2. *What are the distributions of features like age, BMI, and glucose level?*
   - 📊 Visualized using histograms and boxplots.
   - 🔍 Identified outliers and missing values.

3. *How are stroke cases related to categorical variables (gender, work type, marital status, etc.)?*
   - 📈 Analyzed using countplots and pie charts.

4. **What are the most correlated features with the target variable stroke?**
   - 🧠 Used correlation heatmap after encoding to select important features.

5. *How do the models perform based on key classification metrics?*
   - ✔ Compared multiple models using Accuracy, Precision, Recall, and F1-Score.

---

## 🛠 Preprocessing Steps

- ✅ Handled missing values in bmi
- ✅ Detected and treated outliers using boxplots
- ✅ Encoded categorical features with LabelEncoder
- ✅ Feature selection using correlation analysis
- ✅ Feature scaling using StandardScaler
- ✅ Balanced the dataset using *SMOTE*

---

## 📈 Models Used

- Naive Bayes
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression

---

## ✅ Evaluation Metrics

Each model was evaluated based on the following metrics:

- *Accuracy*
- *Precision*
- *Recall*
- *F1-Score*

📊 Results were displayed in:
- Tabular format
- Bar charts for visual comparison
- Confusion matrix and classification report (especially for the best model)

---

## ⭐ Best Model

The *Random Forest* classifier showed the best performance and was selected for final evaluation:

- High accuracy
- Balanced precision and recall
- Good F1-Score

---

## 📷 Visualizations

- 🔥 Correlation heatmap
- 📊 Countplots and pie charts for categorical variables
- 📉 Boxplots for numeric features (age, glucose, BMI)
- 📦 Stroke distribution (before and after SMOTE)
- 📋 Confusion Matrix and Classification Report
- 📈 Bar plots comparing model performance

---

## 📚 Requirements

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn