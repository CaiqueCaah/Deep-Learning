import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

base = pd.read_csv('iris.csv')

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    
    classificador = Sequential()
    
    classificador.add(Dense(units = 5, activation = 'sigmoid', 
                        kernel_initializer = 'identity', input_dim = 4))
    #20% DOS NEURONIOS SERAM ZERADOS PARA EVITAR OVERFITING
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units = 5, activation = 'sigmoid', 
                        kernel_initializer = 'identity'))
    #20% DA CAMADA ESCONDIDA SERAM ZERADOS PARA EVITAR OVERFITING
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = 'RMSprop', loss = 'sparse_categorical_crossentropy', 
                          metrics = ['accuracy'])

    return classificador

#CRIANDO O CLASSIFICADOR KERAS USANDO A FUNÇÃO
classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 100,
                                batch_size = 10)

#VALIDAÇÃO CRUZADA
#CV É A QUANTIDADE DE TESTES
resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')
media = resultados.mean()
#DESVIO PADRÃO
#MAIOR VALOR = TENDENCIA DE OVERFITING
desvio = resultados.std()
