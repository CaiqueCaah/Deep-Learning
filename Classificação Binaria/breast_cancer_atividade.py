# -*- coding: utf-8 -*-
"""
Created on Sat May 16 01:40:16 2020

@author: caiqsilv
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units = 32, activation = 'relu', 
                        kernel_initializer = 'normal', input_dim = 30))
    classificador.add(Dropout(0.3))
    
    classificador.add(Dense(units = 32, activation = 'relu', 
                        kernel_initializer = 'normal'))
    classificador.add(Dropout(0.3))

    classificador.add(Dense(units = 32, activation = 'relu', 
                        kernel_initializer = 'normal'))
    classificador.add(Dropout(0.3))
    
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
    
    return classificador

#CRIANDO O CLASSIFICADOR KERAS USANDO A FUNÇÃO
classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 150,
                                batch_size = 10)
#CV É A QUANTIDADE DE TESTES
resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')
media = resultados.mean()
#DESVIO PADRÃO
#MAIOR VALOR = TENDENCIA DE OVERFITING
desvio = resultados.std()
