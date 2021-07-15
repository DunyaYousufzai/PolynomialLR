import pandas as pd
import numpy as np
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class polynomial_linear_regression:
    def __init__(self, file):
        self.file = file

    def data_selection(self, a, b,c):
       global real_x, real_y
       data = pd.read_csv(self.file)
       real_x = data.iloc[:,a:b].values
       real_y = data.iloc[:,c].values
         #  you need 2d array
       real_x =real_x.reshape(-1,1)
       real_y =real_y.reshape(-1,1)

    def trainig(self, k, ts):
        global linear_reg, polynomial_reg, real_x_poly,   training_x, testing_x,training_y,testing_y
        polynomial_reg = PolynomialFeatures(degree=k)
        
        training_x, testing_x,training_y,testing_y = train_test_split(real_x, real_y,test_size = ts, random_state = 0)
        real_x_poly = polynomial_reg.fit_transform(training_x)
        polynomial_reg.fit(real_x_poly, training_y)
        linear_reg = LinearRegression()
        linear_reg.fit(real_x_poly, training_y)
    
    def plot(self, title,xlabel,ylabel):
        plt.scatter(training_x, training_y, color= "red")
        plt.plot(training_x, linear_reg.predict(polynomial_reg.fit_transform(training_x)), color = "blue")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    
    def predict(self, f):
        x = linear_reg.predict(polynomial_reg.fit_transform([[f]]))
        print(x)

poly = polynomial_linear_regression("polynomial_linear_reg.csv")
poly.data_selection(1,2,2)
poly.trainig(4, 0.2)
poly.plot("polynomial Model", "position","salary")
poly.predict(7)