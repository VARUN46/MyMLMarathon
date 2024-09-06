import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#training data about how much i will charge and how much distance it covers based on recorded values
charging_hours = np.array([1,2,3,4,5]).reshape(-1,1) #independent variable = value that is trying to calculate dependent value
distance_covered = np.array([5,7,13,20,23]) #dependent variable = value we are trying to predict

#create instance of model
model = LinearRegression()

#fit model
model.fit(charging_hours, distance_covered)

#predict value
charged_hours = 3
prediction_input = np.array([charged_hours]).reshape(-1,1)
predicted_output = model.predict(prediction_input)
print(predicted_output)