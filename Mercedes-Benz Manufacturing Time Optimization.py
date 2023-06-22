#!/usr/bin/env python
# coding: utf-8

# In[1]:


# We are using combinations of nameless features in a Mercedes-Benz manufacturing process to predict and optimize the value of 'y', the amount of time spent in the factory.

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# Read the train CSV file into a DataFrame
df_train = pd.read_csv('/Users/wkammerait/Desktop/ML Data Sets/Mercedes Data Sets/train mercedes.csv')

df_test = pd.read_csv('/Users/wkammerait/Desktop/ML Data Sets/Mercedes Data Sets/test mercedes.csv')

df_train.head()


# In[2]:


df_test.head()


# In[3]:


# We could drop 0-variance columns here. Instead we are dropping low-variance columns.

# Calculate the variance for each column
variances = df_train.var()

# Get the column names with zero variance
zero_variance_columns = variances[variances == 0].index.tolist()

# Drop columns with zero variance -- drop these from test data as well.
df_train = df_train.drop(columns=zero_variance_columns)
df_test = df_test.drop(columns=zero_variance_columns)

# Display the dropped columns
print("Dropped columns with == 0 variance:")
print(zero_variance_columns)


# In[5]:


# We can double check 0 variances here if necessary.
variances


# In[6]:


# Calculate the variance for each column
variances = df_train.var()

# Reset index and convert to DataFrame
variances_df = variances.reset_index()

# Rename the columns of the DataFrame
variances_df.columns = ['Column', 'Variance']

# Display the column names and variances
print(variances_df)


# In[7]:


# We see that there are no null values in the training data.
null_counts = df_train.isnull().sum()

sorted_null_counts = null_counts.sort_values(ascending=False)

print(sorted_null_counts.head(40))


# In[8]:


# We see that there are no null values in the test data.
null_counts = df_test.isnull().sum()

sorted_null_counts = null_counts.sort_values(ascending=False)

print(sorted_null_counts.head(40))


# In[9]:


# Count unique values in the training data.
unique_counts = df_train.nunique()

# Sort the unique counts in descending order
sorted_unique_counts = unique_counts.sort_values(ascending=False)

# Print the unique counts
print(sorted_unique_counts)


# In[10]:


# Count unique values in the training data.
unique_counts = df_test.nunique()

# Sort the unique counts in descending order
sorted_unique_counts = unique_counts.sort_values(ascending=False)

# Print the unique counts
print(sorted_unique_counts)


# In[11]:


# Apply label encoder.
class CustomEncoder:
    def __init__(self):
        self.mapping = {}

    def fit_transform(self, data):
        codes, uniques = pd.factorize(data)
        self.mapping = {unique: code for code, unique in enumerate(uniques)}
        return codes

    def transform(self, data):
        return [self.mapping.get(x, -1) for x in data]

encoders = {}
rows_with_new_values = set()
common_columns = set(df_train.columns) & set(df_test.columns)

for column in df_train.columns:
    if column in common_columns and df_train[column].dtype == 'object':
        encoders[column] = CustomEncoder()
        df_train[column] = encoders[column].fit_transform(df_train[column])

for column in df_test.columns:
    if column in common_columns and df_test[column].dtype == 'object':
        transformed_data = encoders[column].transform(df_test[column])
        rows_with_new_values.update(i for i, v in enumerate(transformed_data) if v == -1)
        df_test[column] = transformed_data

print(f"Number of unique rows in the test set with new values: {len(rows_with_new_values)}")


# In[12]:


# Check to ensure there are no non-numeric columns remaining
non_numeric_columns = df_test.select_dtypes(exclude='number').columns
print(non_numeric_columns)

df_train.head()


# In[13]:


df_test.head()


# In[14]:


columns_only_in_test = set(df_test.columns) - set(df_train.columns)
num_columns_only_in_test = len(columns_only_in_test)
print("Number of columns in df_test not present in df_train:", num_columns_only_in_test)


# In[15]:


# Perform dimensionality reduction, first determining the minimum number of components to explain 90% of the variance in the target.
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Identify the numerical columns in the DataFrame
numerical_columns = df_train.select_dtypes(include=['float64', 'int64']).columns
numerical_columns = numerical_columns.drop('y')  # Exclude 'y' column

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to the numerical columns and transform them
df_train[numerical_columns] = scaler.fit_transform(df_train[numerical_columns])
df_test[numerical_columns] = scaler.transform(df_test[numerical_columns])

# Separate the features from the target variable
X_train = df_train.drop(columns=['y'])

# Apply PCA
pca = PCA()

# Fit the PCA to the training data and transform it
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(df_test)

# Calculate cumulative explained variance ratio
explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)

# Find the minimum number of components for desired explained variance ratio
desired_variance_ratio = 0.90  # 90% explained variance
min_components = np.argmax(explained_variance_ratio_cumulative >= desired_variance_ratio) + 1

# Print the minimum number of components
print("Minimum number of components for {} explained variance ratio: {}".format(
    desired_variance_ratio, min_components))


# In[16]:


# Perform principal component analysis.

import pandas as pd
from sklearn.decomposition import PCA

# Separate the features from the target variable
X = df_train.drop(columns=['y'])

# Apply PCA with 120 components
n_components = 120
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Print the transformed data
print(X_pca)


# In[18]:


#Run the XGBoost model. This is the block used to create the predicted 'y' values for the test data frame. The other blocks attempted other methods to see if the r^2 score could be improved.

from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score


# Separate the features from the target variable
X = df_train.drop(columns=['y'])
y = df_train['y']

# Apply PCA with 10 components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Set default hyperparameter values
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=1.0
)

# Train the XGBoost model on the training data with cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print("Cross-Validation R-squared Scores:", scores)
print("Mean Cross-Validation R-squared:", scores.mean())

# Fit the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the training data
predictions_train = model.predict(X_train)

# Calculate R-squared on the training data
r2_train = r2_score(y_train, predictions_train)
print("R-squared on Training Data:", r2_train)

# Evaluate the model on the validation data
predictions_val = model.predict(X_val)

# Calculate R-squared on the validation data
r2_val = r2_score(y_val, predictions_val)
print("R-squared on Validation Data:", r2_val)


# In[40]:


#Run GridSearch to determine optimal hyperparameters.

from sklearn.model_selection import GridSearchCV
import numpy as np

# Create a dictionary of hyperparameters and their ranges
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 50, 500],
    'learning_rate': [0.1, 0.01,1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Create the XGBRegressor model
model = xgb.XGBRegressor()

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the corresponding mean cross-validated score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Print the best hyperparameters and the corresponding score
print("Best Hyperparameters:", best_params)
print("Best R-squared Score:", best_score)


# In[28]:


# This block is using L1 regularization. 
model = xgb.XGBRegressor(
    reg_alpha=10,
    learning_rate=0.05,
    subsample=0.8,
    gamma=200,
    n_estimators=100,
    max_depth=15,
    colsample_bytree=1.0)
model.fit(X_train, y_train)

# Evaluate the model on the training data
predictions_train = model.predict(X_train)
mse_train = mean_squared_error(y_train, predictions_train)
r2_train = r2_score(y_train, predictions_train)

# Evaluate the model on the validation data
predictions_val = model.predict(X_val)
mse_val = mean_squared_error(y_val, predictions_val)
r2_val = r2_score(y_val, predictions_val)

print("Mean Squared Error on Training Data:", mse_train)
print("R-squared on Training Data:", r2_train)
print("Mean Squared Error on Validation Data:", mse_val)
print("R-squared on Validation Data:", r2_val)

from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=8, scoring='r2')

# Print the cross-validation scores
print("\nCross-validation scores:", cv_scores)
print("Mean R-squared:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())


# In[32]:


# Try L2 regularization to see if the performance improves. Using hyperparameter tuning and L2 regularization this model had the strongest cross-validation score and there was a 2.9 pt increase in the r^2 score on the validation data compared with the XGBoost model with no regularization.
model = xgb.XGBRegressor(
    reg_lambda=10,
    n_estimators=100,
    max_depth=15,
    learning_rate=0.05,
    subsample=0.8,
    gamma = 200,
    colsample_bytree=1.0,
    )  # experiment with different values for lambda
model.fit(X_train, y_train)

# Evaluate the model on the validation data
predictions = model.predict(X_val)

# Calculate R-squared score
r2_score_val = r2_score(y_val, predictions)
print("R-squared on Validation Data:", r2_score_val)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_val, predictions)
print("Mean Squared Error on Validation Data:", mse)

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=8, scoring='r2')

# Print the cross-validation scores
print("\nCross-validation scores:", cv_scores)
print("Mean R-squared:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())


# In[52]:


# L2 regularization provided the strongest r^2 score, so these are the values we will use for the predicted targets.
# Separate the features from the target variable in the test set
X_test = df_test

# Apply the same PCA transformation as in the training set
X_test_pca = pca.transform(X_test)

# Predict the target variable for the test set
predictions_test = model.predict(X_test_pca)

# Create a DataFrame with the predicted target values
df_predictions_test = pd.DataFrame({'y': predictions_test})

# Save the predictions to a CSV file
df_predictions_test.to_csv('predictions.csv', index=False)

# Print the DataFrame with the predicted target values
print(df_predictions_test)


# In[ ]:


# If you need to save these predictions to a CSV file, you can do this:
pd.DataFrame(predictions, columns=['Predicted_y']).to_csv('predictions.csv', index=False)


# In[29]:


# Run a linear regression for comparison. We see that the R-squared on the validation data is much lower, so this would not be used.
from sklearn.linear_model import LinearRegression

# Initialize the model
linear_model = LinearRegression()

# Fit the model on training data
linear_model.fit(X_train, y_train)

# Make predictions on the validation set
predictions_lr = linear_model.predict(X_val)

from sklearn.metrics import mean_squared_error, r2_score

# Make predictions on the training set
predictions_train = model.predict(X_train)

# Compute R-squared score on the training data
r2_score_train = r2_score(y_train, predictions_train)
print("R-squared on Training Data:", r2_score_train)

# Compute Mean Squared Error (MSE) on the training data
mse_train = mean_squared_error(y_train, predictions_train)
print("Mean Squared Error on Training Data:", mse_train)

# Evaluate the model on the validation data
r2_score_val_lr = r2_score(y_val, predictions_lr)
print("R-squared on Validation Data (Linear Regression):", r2_score_val_lr)

# Calculate Mean Squared Error (MSE)
mse_lr = mean_squared_error(y_val, predictions_lr)
print("Mean Squared Error on Validation Data (Linear Regression):", mse_lr)



# Normalization is not included here because it did not materially change the R-squared on the validation data.


# In[46]:


# Appendix -- view distributions of each column. 

import matplotlib.pyplot as plt

# Assuming selected_columns is your list of the 20 column names
selected_columns = df_train.columns[:20]  # Replace this line with your actual columns

fig, axs = plt.subplots(10, 2, figsize=(15, 30))  # Adjust the size as needed
axs = axs.ravel()

for i, column in enumerate(selected_columns):
    axs[i].scatter(df_train[column], df_train['y'])
    axs[i].set_title(f'{column} vs. y')
    axs[i].set_xlabel(column)
    axs[i].set_ylabel('y')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




