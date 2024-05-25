import pandas as pd
from sklearn import linear_model

df=pd.read_csv("data.csv")

X=df[['Weight','Volume']]
y=df[['CO2']]

regr=linear_model.LinearRegression()
regr.fit(X,y)

#*predicting
# predictedCO2=regr.predict([[2300,1300]])
predictedCO2=regr.predict([[3300,1300]])

print(predictedCO2)
# print(regr.coef_)