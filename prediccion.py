#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 20:47:49 2020

@author: Aronnvega
"""

# Parte 4 Creacion de la predicion 
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura = 150, 150
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'
cnn=load_model(modelo)
cnn.load_weights(pesos)

def predict(file):
    x=load_img(file, target_size=(longitud, altura))
    x=img_to_array(x)
    x=np.expand_dims(x, axis=0)
    arreglo=cnn.predict(x) ## Esto seria un arreglo de 2 dimensiones[[1,0,0]]
    resultado=arreglo[0] ## este nos trae el resultado de nuestra prediccion [[0,0,1]]
    respuesta= np.argmax(resultado) ## la respuesta se sera 2
    if respuesta==0:
        print('Perro')
    elif respuesta==1:
        print('Gato')
    elif respuesta==2:
        print('Gorilla')
    return respuesta  
            
predict('./test/perro.jpeg')