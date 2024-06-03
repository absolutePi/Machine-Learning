import numpy as np
from sklearn import linear_model

#!predicting if a tumor is malignant or benign
#*X is the size of a tumor in mm
X=np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
#*y is whether tumor is cancerous or not
y=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr=linear_model.LogisticRegression()
logr.fit(X,y)

#!prediction
# predicted=logr.predict(np.array([3.46]).reshape(-1,1))
# print(predicted)

log_odds=logr.coef_
odds=np.exp(log_odds)
#*if the size increases by 1, the chance of the tumor begin cancerous increases by 4x(4.03557295)
# print(odds)

def logit2prob(logr,X):
    log_odds=logr.coef_*X+logr.intercept_
    odds=np.exp(log_odds)
    probability=odds/(1+odds)
    return(probability)
print(logit2prob(logr,X))