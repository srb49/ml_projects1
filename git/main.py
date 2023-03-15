#Salary predictin based on years of experience
#Linear regression model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

dataset= pd.read_csv('Salary.csv')
print(dataset)

x = dataset.iloc[:, :1].values
#print(x)
y = dataset.iloc[:, 1:].values
#print(y)


fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.scatter(x, y, color='r')
'''plt.xlabel(x)
plt.ylabel(y)'''
#plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

print(y_pred)
print(y_test)

plt.scatter(x, y, color='r')
plt.plot(x, regressor.predict(x), color='blue')
#plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x)
regressor.fit(x_poly, y)

plt.scatter(x,y, color='r')
plt.plot(x, regressor.predict(poly.fit_transform(x)), color='black')
plt.show()