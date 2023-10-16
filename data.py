from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

import pandas as pd


# Load the data

data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['PRICE'] = data.target

print(df.head())

# Split the data
X = df.drop('PRICE', axis=1)
y = df['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model - using a linear regression

# TODO: figure out how to use scikit learn to make a linear regression model
