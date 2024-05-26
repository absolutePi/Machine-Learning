import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()

df=pd.read_csv("data.csv")

X=df[['Weight','Volume']]
y=df['CO2']

scaledX=scale.fit_transform(X)

regr=linear_model.LinearRegression()
regr.fit(scaledX,y)

scaled=scale.transform([[2300,1.3]])
predictedCO2=regr.predict([scaled[0]])
# print(scaledX)
print(predictedCO2)
