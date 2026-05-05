import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv('C:\\Users\\Bhavitha\\Desktop\\MEACHINE LEARNING\\test_scores.csv')
x = data[['math']]
y = data['cs']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model =LinearRegression()
model.fit(x_train, y_train)

joblib.dump(model, 'model.pkl')
print("Model saved successfully!")

loaded_model = joblib.load('model.pkl')

new_data = np.array([[85]])

predicted_score = loaded_model.predict(new_data)
print("Predicted score for math score 85: ", predicted_score[0])


# output:
# Model saved successfully!
# Predicted score for math score 85:  88.12831430230307
