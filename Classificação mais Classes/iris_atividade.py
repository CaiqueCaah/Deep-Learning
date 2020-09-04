import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

base = pd.read_csv('iris.csv')

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    
    classificador = Sequential()
    
    classificador.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer, input_dim = 4))
    #20% DOS NEURONIOS SERAM ZERADOS PARA EVITAR OVERFITING
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer))
    #20% DA CAMADA ESCONDIDA SERAM ZERADOS PARA EVITAR OVERFITING
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = optimizer, loss = loos, 
                          metrics = ['accuracy'])

    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [10, 30, 64],
              
              'epochs': [5],
              
              'optimizer': ['adam', 'sgd', 'RMSprop', 'adadelta', 'adagrad', 
                            'adamax'],
                            
              'loos': ['sparse_categorical_crossentropy'],
              
              'kernel_initializer': ['random_uniform', 'normal', 'random_normal'],
              
              'activation': ['relu', 'sigmoid', 'softmax'],
              
              'neurons': [8]}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           cv = 5)

grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_