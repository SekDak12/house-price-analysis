#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("train.csv")


# In[ ]:


df.head()


# In[ ]:


# Check for missing values in each column
missing_values = df.isnull().sum().sort_values(ascending=False)

# Filter out the columns that have missing values
missing_values = missing_values[missing_values > 0]

missing_values


# In[24]:


# Define the columns to drop
columns_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 
                   'GarageYrBlt', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 
                   'BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 
                   'MasVnrArea', 'MasVnrType', 'Electrical']

# Drop the columns
df = df.drop(columns=columns_to_drop)
df.head()


# In[ ]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

# Identify the categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Apply one-hot encoding to the categorical columns
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), categorical_columns),
    remainder='passthrough'
)

# Define the target variable
y = df['SalePrice']

# Define the feature set
X = df.drop(columns=['SalePrice'])

# Create a pipeline that first one-hot encodes the data and then applies RFE
model = make_pipeline(
    preprocessor,
    RFE(estimator=RandomForestRegressor(), n_features_to_select=20)
)

# Fit the model
model.fit(X, y)

# Get the feature ranking
ranking = model.named_steps['rfe'].ranking_

# Get the names of the columns after one-hot encoding
feature_names = model.named_steps['columntransformer'].get_feature_names_out()

# Create a DataFrame that associates each feature with its ranking
features_rfe = pd.DataFrame({'Feature': feature_names, 'Ranking': ranking})

# Show the top 10 features
top_features_rfe = features_rfe[features_rfe.Ranking == 1]
top_features_rfe


# In[47]:


# Use the fitted model to transform the feature set
X = df.drop(columns=['SalePrice'])

X_transformed = model_rfe.transform(X)


# Train a random forest model using the transformed feature set

rf = RandomForestRegressor(n_estimators=100, random_state=4002)

rf.fit(X_transformed, y)



rf


# In[64]:


# Load the new test dataset
test_df = pd.read_csv('test.csv')

# Select the same features as in the training set
test_X = test_df

test_X_puro = test_X.drop(columns=columns_to_drop)
test_X_filled = test_X_puro.fillna(0)


# Use the RFE model to transform the imputed test feature set
test_X_transformed = model_rfe.transform(test_X_filled)

# Use the trained random forest model to make predictions on the test set
test_y_pred = rf.predict(test_X_transformed)

# Create a DataFrame with the IDs and the predicted prices
predictions_df = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': test_y_pred
})

predictions_df.head()


# In[61]:


b= test_X_puro.isnull().sum().sort_values(ascending=False)

# Filter out the columns that have missing values
missing_values2 = b[b > 0]

missing_values2


# In[60]:


a= X.isnull().sum().sort_values(ascending=False)

# Filter out the columns that have missing values
missing_values23 = a[a > 0]

missing_values23


# In[ ]:




