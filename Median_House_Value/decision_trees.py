import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('housing.csv')
data.drop('ocean_proximity', axis = 1, inplace = True)
data = data.dropna()

x = data.drop("median_house_value", axis = 1)
y = data['median_house_value']

#%%

##### Profundidade ilimitada ####

xtr, xval, ytr, yval = train_test_split(x,y, test_size = 0.5, random_state = 0)

t = DecisionTreeRegressor()
t.fit(xtr, ytr)

p = t.predict(xval) #previsão

print(np.sqrt(mean_squared_error(yval, p)))

# plot_tree(t) #arvore muito grande, overfitting; vai pegando até nós com 1 exemplo só (overfit extremo)
# Para evitar esse overfitting, podemos limitar a profundidade da árvore

#%%

#### Limitando profundidade ####

xtr, xval, ytr, yval = train_test_split(x,y, test_size = 0.5, random_state = 0)

t = DecisionTreeRegressor(max_depth = 1)
t.fit(xtr, ytr)
p = t.predict(xval) #previsão
print(np.sqrt(mean_squared_error(yval, p))) # erro agora é menor, mas não há overfitting; podemos melhorar com mais profundidade

plot_tree(t, feature_names = xtr.columns) # value é a previsão do valor pro caso

#%%

#### Descobrindo o max_depth de erro minimo no teste ####

xtr, xval, ytr, yval = train_test_split(x, y, test_size = 0.5, random_state = 0)
i = 1
erro_min = np.inf
i_min = 1
while(i < 1000):
    t = DecisionTreeRegressor(max_depth = i, random_state = 0)
    t.fit(xtr, ytr)
    p = t.predict(xval)
    erro = np.sqrt(mean_squared_error(yval, p))
    if erro < erro_min:
        erro_min = erro
        i_min = i
    i = i + 1
print(i_min) # no exemplo, deu i_min = 9
    
#%%

#### Usando min_samples_leaf ####
# Uma alternativa a setar diretamente o max_depth

xtr, xval, ytr, yval = train_test_split(x, y, test_size = 0.5, random_state = 0)
t = DecisionTreeRegressor(min_samples_leaf = 10, random_state = 0)
# posso fazer um loop também pra ver o min_sample_leaf de menor erro

t.fit(xtr, ytr)
p = t.predict(xval)
print(np.sqrt(mean_squared_error(yval, p)))










