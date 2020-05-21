import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer, input_dim = 30))
    #20% DOS NEURONIOS SERAM ZERADOS PARA EVITAR OVERFITING
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer))
    #20% DA CAMADA ESCONDIDA SERAM ZERADOS PARA EVITAR OVERFITING
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    classificador.compile(optimizer = optimizer, loss = loos,
                      metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [10, 30, 64],
              
              'epochs': [50],
              
              'optimizer': ['adam', 'sgd', 'RMSprop', 'adadelta', 'adagrad', 
                            'adamax'z],
              
              'loos': ['binary_crossentropy', 'hinge'],
              
              'kernel_initializer': ['random_uniform', 'normal', 'random_normal', 
                                     'truncated_normal', 'zeros', 'ones', 
                                     'glorot_normal', 'glorot_uniform', 'identity'],
              
              'activation': ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 
                             'tanh', 'selu', 'elu', 'exponential'],
              
              'neurons': [16, 8]}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 5)

grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_