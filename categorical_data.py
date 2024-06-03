import pandas as pd
from sklearn import linear_model

cars=pd.read_csv('data.csv')
ohe_cars=pd.get_dummies(cars[['Car']])

# print(cars.to_string())
# print(ohe_cars.to_string())

X=pd.concat([cars[['Volume','Weight']],ohe_cars],axis=1)
y=cars['CO2']

regr=linear_model.LinearRegression()
regr.fit(X,y)
#!predicting the CO2 emission of a Volvo with 2300kg weight and 1300cm3 volume
predictedCO2=regr.predict([[2300, 1300,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])
# print(predictedCO2)
#*Dummifying
colors=pd.DataFrame({'color':['blue','red','green']})
# print(colors)
dummies=pd.get_dummies(colors,drop_first=True)
dummies['color']=colors['color']
print(dummies)
