import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df=pd.read_csv("data02.csv")

d={'UK':0,'USA':1,'N':2}
df['Nationality']=df['Nationality'].map(d)
d={'YES':1,'NO':0}
df['Go']=df['Go'].map(d)
# print(df)

features=['Age','Experience','Rank','Nationality']

X=df[features]
y=df['Go']

# print(X)
# print(y)
dtree=DecisionTreeClassifier()
dtree=dtree.fit(X,y)

tree.plot_tree(dtree,feature_names=features)
#!predicting
# print(dtree.predict([[40,10,7,1]]))
print(dtree.predict([[40,10,6,1]]))