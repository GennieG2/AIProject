import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# fetch dataset 
chronic_kidney_disease = fetch_ucirepo(id=336) 
  
# data (as pandas dataframes) 
X = chronic_kidney_disease.data.features 
y = chronic_kidney_disease.data.targets 

X.replace('?', np.nan, inplace=True)

for col in X.columns:
    try:
        X.loc[:, col] = pd.to_numeric(X[col])
    except:
        pass

# Identify categorical columns (usually 'object' dtype)
categorical_cols = X.select_dtypes(include='object').columns.tolist()
print("Categorical columns:", categorical_cols)

# Define which columns are ordinal and which are nominal (you should know this from domain knowledge)
# Example (adjust according to your dataset):
ordinal_cols = ['severity', 'stage']  # Add your ordinal columns here
nominal_cols = [col for col in categorical_cols if col not in ordinal_cols]

# Step 1: Apply Label Encoding for Ordinal Variables
label_encoder = LabelEncoder()
for col in ordinal_cols:
    X[col] = label_encoder.fit_transform(X[col])

# Step 2: Apply One-Hot Encoding for Nominal Variables
X_encoded = pd.get_dummies(X, columns=nominal_cols, drop_first=True)

# Step 3: Encode the target variable (y) if it's categorical using Label Encoding
# (This step is only necessary if y is categorical, not continuous)
if y.dtype == 'object':
    y_encoded = label_encoder.fit_transform(y)
else:
    y_encoded = y  # If y is continuous, no encoding needed

# Output the encoded features and target
print("\nEncoded features:")
print(X_encoded.head())

print("\nEncoded target:")
print(y_encoded[:10])  # Show the first 10 values of the encoded target

# Check where NaNs are in the data
# print("NaN values in the dataset:")
# print(X.isna().sum())
  
# Print variable information 
# print(chronic_kidney_disease.variables) 