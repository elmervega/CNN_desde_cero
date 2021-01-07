#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 15:35:36 2020

@author: Aronnvega
"""

import sys 
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as k
from tensorflow.python.keras import applications

vgg = applications.vgg16.VGG16()

vgg.summary()

# Aqui le agregamos cada una de las capa al modelo cnn que creamos
cnn=Sequential()
for capa in vgg.layers:
    cnn.add(capa)

# Aqui ya tenemos la misma estructura de VGG16
cnn.summary()

# Eliminaremos la ultima capa, la cual es la prediccion
cnn.pop()

# Verificamos que la ultima capa de prediccion haya sido eliminada
cnn.summary()

# esto hara que por cada capa de entrenamiento de la anterior no aprendan con los pesos
for layer in cnn.layers:
    layer.trainable=False
    
# Queremos utilizar tres clases (Perros, Gatos y Gorillas)   
cnn.add(Dense(3, activation='softmax'))

cnn.summary()

# Aqui creamos una funcion para la red de Convolucion
def modelo():
    vgg=applications.vgg16.VGG16()
    cnn=Sequential()
    
    for capa in vgg.layers:
        cnn.add(capa)
    cnn.layers.pop()
    for layer in cnn.layers:
        layer.trainable=False
    cnn.add(Dense(3, activation='softmax'))

    return cnn

# Aqui llamamos a toda la clase anterior
k.clear_session()

data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

epocas=20
longitud, altura = 224, 224
batch_size = 32
pasos = 1000
validation_steps = 300
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3,3)
tamano_filtro2 = (2,2)
tamano_pool = (2,2)
clases = 3
lr = 0.0004

## Preparamos las imagenes
entrenamiento_datagen =  ImageDataGenerator(
    rescale=1. /255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
    )

test_datagen = ImageDataGenerator(rescale=1. /255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')


# Creacion de la red VGG16

cnn=modelo()

cnn.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
            
cnn.fit(
        entrenamiento_generador,
        steps_per_epoch=pasos,
        epochs=epocas,
        validation_data=validacion_generador,
        validation_steps=validation_steps)

target_dir='./ modelo'
if not os.path.exists(target_dir):
    os.makedir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')

### Grafico Historico de la RNC
%matplotlib inline 
import matplotlib.pyplot as plt 
import numpy as np 
plt.figure(0)  
plt.plot(cnn_history.history['accuracy'],'r')  
plt.plot(cnn_history.history['val_accuracy'],'g')  
plt.xticks(np.arange(0, 50, 1.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy")  
plt.legend(['train','validation'])

plt.figure(1)  
plt.plot(cnn_history.history['loss'],'r')  
plt.plot(cnn_history.history['val_loss'],'g')  
plt.xticks(np.arange(0, 50, 1.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss")  
plt.legend(['train','validation'])

plt.show()