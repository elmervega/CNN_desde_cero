#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 11:40:54 2020

@author: Aronnvega
"""
import pandas as pd
import sys 
import os 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator # Para Preprocesar
from tensorflow.python.keras import optimizers # Optimizar el Algoritmo
from tensorflow.python.keras.models import Sequential # Para realizar que cada una de las capas estan en orden
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation 
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D # Nuestras capas para convuliciones
from tensorflow.python.keras import backend as k # Por si esta corriendo alguna aplicacion de keras por detras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.python.keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
from mlxtend.evaluate import confusion_matrix
import matplotlib.image as mpimg

k.clear_session()

# Aqui colocaremos el directorio de nuestra RNC
data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'
directorio_test = './data/pruebas'

## Parametros de nuestra RNC

epocas = 25
altura, longitud = 200,200 # Aqui asignamos los pixeles
batch_size = 32 # Es la cantidad de imagenes que le enviaremos al ordenador a procesar
#pasos = 1000 # Aqui seran el numeros de veces que se procesaran los datos en las epocas
#pasos_validacion = 200 # Se recorreran 200 pasos dentro de nuestra carpeta de validacion para aprender
filtrosConv1 = 16
filtrosConv2 = 32
filtrosConv3 = 64
filtrosConv4 = 128
tamano_filtro1 = (3,3)
tamano_filtro2 = (2,2)
tamano_filtro3 = (2,2)
tamano_filtro4 = (1,1)
tamano_pool = (2,2)
clases = 3 # En esta es para clasificar en la cantidad de objetos que deseamos 
lr = 0.0005 # Ajustes para acercarse a una solucion optima

# Pre_procesamieno de imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range = 0.3, # Para inclinar 
    zoom_range = 0.3, # Alguna para hacerle zoom
    horizontal_flip = True # Para invertir la imagen
    )

validacion_datagen = ImageDataGenerator(
    rescale = 1./255
    )

image_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode= 'categorical'
    )
image_validacion =  validacion_datagen.flow_from_directory(
    data_validacion,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
    )

print(image_entrenamiento.class_indices)

pasos_entrenamiento = image_entrenamiento.n//image_entrenamiento.batch_size
pasos_validacion = image_validacion.n//image_validacion.batch_size


## Creacion de la RNC

cnn = Sequential()

cnn.add(Conv2D(filtrosConv1, tamano_filtro1, padding='same', 
               input_shape = (altura, longitud,3), activation= 'relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Conv2D(filtrosConv2, tamano_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Conv2D(filtrosConv3, tamano_filtro3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Conv2D(filtrosConv4, tamano_filtro4, padding='same', activation='elu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu')) # Aqui se asigna la cantidad de neuronas que se usaran
cnn.add(Dropout(0.3)) #  Lo que realiza el dropout es apagar el 50% de la neuronas para que no aprenda siempre el mismo camino
cnn.add(Dense(clases, activation='softmax'))

# Parametros para optimizar nuestros algoritmos
cnn.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) 
# categorical_crossentropy : Es el que nos dira que tan bien o mal va nuestra RNC
# accuracy : ES lo que nos dice que tambien esta aprendiendo nuestra RNC
print(cnn.summary())
## Llamamos la funcion Callback para que si el entrenamiento pare si no mejora

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose = 1, min_delta=1e-4)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, min_delta=1e-4)
callbacks_list = [early_stop, reduce_lr]

cnn_history = cnn.fit(image_entrenamiento,
                        epochs = epocas,
                        validation_data = image_validacion,
                        validation_steps=pasos_validacion,
                        callbacks=callbacks_list
                        )



### Grafico Historico de la RNC

import matplotlib.pyplot as plt 
import numpy as np 
plt.figure(0)  
plt.plot(cnn_history.history['accuracy'],'r')  
plt.plot(cnn_history.history['val_accuracy'],'g')  
plt.xticks(np.arange(0, 25, 1.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy")  
plt.legend(['train','validation'])

plt.figure(1)  
plt.plot(cnn_history.history['loss'],'r')  
plt.plot(cnn_history.history['val_loss'],'g')  
plt.xticks(np.arange(0, 25, 1.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss")  
plt.legend(['train','validation'])

plt.show()

## test de la RNC a la carpeta con la que haremos predicciones

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    directorio_test,
    target_size=(altura, longitud),
    color_mode='rgb',
    batch_size=1,
    
    )
print(test_generator.class_indices)

## Realizacion de prediccion 
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
test_generator.reset() 
pred=cnn.predict(test_generator, steps=STEP_SIZE_TEST, verbose=0)

pred_class_indices = np.argmax(pred,axis=1)

print(pred_class_indices)
print(type(pred_class_indices))

labels = (image_entrenamiento.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in pred_class_indices]

## Comparacion de la clasfisicaion de la RNC

filenames = test_generator.filenames
results = pd.DataFrame({"Filename":filenames,
                        "Predictions":predictions})
results.to_csv("results1.csv",index=False)

real_class_indices=[]
for i in range (0, len(filenames)):
    your_path = filenames[i]
    path_list = your_path.split(os.sep)
    if ("Gato" in path_list[1]) :
        real_class_indices.append(0)
    if ("Gorilla" in path_list[1]) :
        real_class_indices.append(1)
    if ("Perro" in path_list[1]) :
        real_class_indices.append(2)

print(real_class_indices)
print(len(real_class_indices))
real_class_indices = np.array(real_class_indices)
print(type(real_class_indices))


 



    
    