from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#gerando massa de dados
x, y = make_regression(n_samples=200, n_features=1, noise=30)
# print(x, y)

#mostrando gráfico
def show_data(x,y):
    plt.scatter(x,y)
    return plt.show()

# show_data(x, y)

#Creating the model
model = LinearRegression()
# model.fit(x,y)
# intercept = model.intercept_
# coef = model.coef_


#Printing the result
# plt.scatter(x, y)
# xreg = np.arange(-5, 5, 1)
# plt.plot(xreg, coef*xreg - intercept, color='red') #Creating the regression line
# # plt.show()

#Splitting the data to build R2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, shuffle=True)
model.fit(x_train, y_train)

#RESULT
result = model.score(x_test, y_test)
print(result)


#Printing the result
intercept = model.intercept_
coef = model.coef_
plt.scatter(x_test, y_test)
xreg = np.arange(-5, 5, 1)
plt.plot(xreg, coef*xreg - intercept, color='red') #Creating the regression line
plt.show()
