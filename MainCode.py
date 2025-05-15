# Task 3: Cuisine Classification using Machine Learning
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("Dataset .csv")

# Select necessary columns
df = df[['Cuisines', 'City', 'Price range', 'Has Online delivery', 'Votes']]

# Drop missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Handle multi-label cuisines: keep only first cuisine
df['Cuisines'] = df['Cuisines'].apply(lambda x: x.split(',')[0].strip())

# Encode categorical features
le_city = LabelEncoder()
df['City'] = le_city.fit_transform(df['City'])

le_delivery = LabelEncoder()
df['Has Online delivery'] = le_delivery.fit_transform(df['Has Online delivery'])

le_cuisine = LabelEncoder()
df['Cuisine Label'] = le_cuisine.fit_transform(df['Cuisines'])

# Features and target
X = df[['City', 'Price range', 'Has Online delivery', 'Votes']]
y = df['Cuisine Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

unique_labels = np.unique(y_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test,y_pred,labels=unique_labels,target_names=le_cuisine.inverse_transform(unique_labels)))

#Optional: Map back prediction to cuisine names
df_results = X_test.copy()
df_results['Actual Cuisine'] = le_cuisine.inverse_transform(y_test)
df_results['Predicted Cuisine'] = le_cuisine.inverse_transform(y_pred)
print(df_results.head())
