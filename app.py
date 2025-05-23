import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Fetch dataset 
chronic_kidney_disease = fetch_ucirepo(id=336) 

# Extract data
X = chronic_kidney_disease.data.features.copy()
y = chronic_kidney_disease.data.targets.copy()

# Clean missing values
X.replace('?', np.nan, inplace=True)

for col in X.columns:
    try:
        X.loc[:, col] = pd.to_numeric(X[col])
    except:
        pass

# Identify categorical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
print("Categorical columns:", categorical_cols)

# Define ordinal and nominal columns
ordinal_cols = ['sg', 'al', 'su', 'appet', 'pe', 'ane']
nominal_cols = [col for col in categorical_cols if col not in ordinal_cols]

# Label encode ordinal variables
label_encoder = LabelEncoder()
for col in ordinal_cols:
    X.loc[:, col] = label_encoder.fit_transform(X[col])

y_encoded = label_encoder.fit_transform(y['class'])
print("\nClass distribution after cleaning:", np.unique(y_encoded, return_counts=True))

# Strip whitespace and standardize the target class
y['class'] = y['class'].str.strip().str.lower()

# Check unique values after cleanup
print("\nCleaned class values:", y['class'].unique())

# Encode target again
y_encoded = label_encoder.fit_transform(y['class'])

# Check distribution again
print("\nClass distribution after cleaning:", np.unique(y_encoded, return_counts=True))


# One-hot encode nominal variables
X_encoded = pd.get_dummies(X, columns=nominal_cols, drop_first=True)

print(y.columns)
y = y['class']

# Encode the target variable
if y.dtype == 'object':
    y_encoded = label_encoder.fit_transform(y)
else:
    y_encoded = y

# Output encoded data
print("\nEncoded features:")
print(X_encoded.head())

print("\nEncoded target:")
print(y_encoded[:10])

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Print the shape of the splits to verify
print("Training features shape:", X_train.shape)
print("Test features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Test target shape:", y_test.shape)

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, filled=True, feature_names=X_encoded.columns, class_names=np.unique(y_encoded).astype(str), rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

print("Class distribution:", np.unique(y_encoded, return_counts=True))
