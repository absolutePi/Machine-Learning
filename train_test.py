import numpy as np
import matplotlib.pyplot as plt
np.random.seed(69)

x=np.random.normal(3,1,100)
y=np.random.normal(150,40,100)/x

train_x=x[:80]
train_y=y[:80]

test_x=x[80:]
test_y=y[80:]
# plt.scatter(x,y)
# plt.scatter(train_x,train_y)
plt.scatter(test_x,test_y)
plt.show()