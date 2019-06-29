# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 11:52:32 2019

@author: DeveshKataria
"""

from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures

#Height weight shoe size
dim = [[180, 72, 10], [165, 55, 8], [167, 65, 8], [170, 67, 7], [172, 60, 9], [160, 70, 9], 
	   [166, 60, 7]]
gender = ['male', 'female', 'female', 'female', 'male', 'male', 'female']


#Decision tree classification
classifier = tree.DecisionTreeClassifier()
classifier.fit(dim, gender)
res = classifier.predict([[160, 77, 9]])
print(res)

#Linear regression
labelencoder = LabelEncoder()
gender = labelencoder.fit_transform(gender)
regressor = LinearRegression()
regressor.fit(dim, gender)
res = regressor.predict([[140, 20, 9]])
print(res)
if res > 0:
	print('male')
else:
	print('female')
	
#Polynomial Regression
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(dim)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, gender)
res = lin_reg_2.predict(poly_reg.fit_transform([[150, 75, 7]]))
if res > 0.4:
	print('male')
else:
	print('female')


