# Task-1-Prediction-using-Supervised-ML
What will be predicted score if a student studies for 9.25 hrs/day?






# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Creating the dataset
data = {'Hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 2.7, 4.8, 3.8, 6.9, 7.8],
        'Scores': [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30, 24, 67, 69, 30, 54, 35, 76, 86]}

df = pd.DataFrame(data)

# Splitting the data into features (X) and target variable (y)
X = df['Hours'].values.reshape(-1, 1)
y = df['Scores'].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating a linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

# Predicting the score for a student studying 9.25 hours
hours_studied = np.array([[9.25]])
predicted_score = model.predict(hours_studied)
print(f'Predicted Score for 9.25 hours of study: {predicted_score[0]}')

# Plotting the regression line
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Scores Obtained')
plt.show()


# Now run your code in jputer Notebook
