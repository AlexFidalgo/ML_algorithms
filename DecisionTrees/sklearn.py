import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('housing.csv')

x = data.drop("median_house_value", axis = 1)
y = data['median_house_value']
xtr, xval, ytr, yval = train_test_split(x,y, test_size = 0.5, random_state = 0)

t = DecisionTreeRegressor()
t.fit(xtr, ytr)

p = t.predict(xval)

print(np.sqrt(mean_squared_error(yval, p)))

