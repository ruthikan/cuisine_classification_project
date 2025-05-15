# 🍽️ Cuisine Classification using Machine Learning

This project was completed as part of my Machine Learning Internship at **Cognizify Technologies** under **Task 3**.  
The objective is to develop a classification model that can predict a restaurant’s cuisine type based on structured features.

---

## 🎯 Objective
To build and evaluate a model that predicts the cuisine of a restaurant using data such as:
- City
- Price Range
- Online Delivery (Yes/No)
- Number of Votes

---

## 📁 Dataset Overview
The dataset includes:
- Cuisines (target variable)
- City (categorical)
- Price Range (1–4)
- Has Online Delivery (Yes/No)
- Votes

---

## 🧹 Preprocessing Steps
- Removed missing values
- Extracted the **primary cuisine** from multiple cuisines
- Encoded categorical columns using LabelEncoder
- Used **stratified train-test split** for balanced classification

---

## 🧠 Model Used
- **Random Forest Classifier** from Scikit-learn
- Chosen for its strong performance on categorical/tabular data

---

## 📊 Evaluation Metrics
- Accuracy
- Precision (weighted)
- Recall (weighted)
- Classification Report

---

## 📈 Results
The model showed moderate accuracy and performed better on high-frequency cuisines.  
Stratification and label encoding improved results and reproducibility.

---

## 💡 Future Enhancements
- Use NLP on descriptions or menus
- Support multi-label classification
- Handle class imbalance with SMOTE or weights
- Try Deep Learning embeddings or transformers

---

## 🛠️ Tools & Libraries
Python, Pandas, Scikit-learn, LabelEncoder, RandomForestClassifier

---

## 🚀 Running the Project
git clone https://github.com/ruthikan/cuisine_classification_project<br>
python MainCode.py<br>

---

## 📝 Author
Ruthika Nalajala<br>
Intern at Cognifyz Technologies<br>
LinkedIn: https://www.linkedin.com/in/ruthika-nalajala-73127628b/<br>

---

## 🙌 Acknowledgment
Thanks to **Cognizify Technologies** for providing an opportunity to apply machine learning to real-world classification problems.
