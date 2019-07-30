"""
Creación del Modelo
"""
import sys
import json
from pathlib import Path
import numpy as np

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras import losses


class Model():
    '''Clase Model.
    
    '''

    def __init__(self):
        '''Inicializador.

            Inicializa el modelo de manera Secuencial usando Keras.
            
            See Also
            --------
            Sequential models : Keras
                (https://keras.io/models/sequential/)

        '''
        
        self.model = Sequential()

    def loadModel(self, model_path):
        '''Compila el modelo desde un fichero.

            Lee un archivo con la configuración del modelo.
            
            Parameters
            ----------
            model_path : string
                Ruta del fichero que contiene la información del modelo.
            
            Returns
            -------
            Secuencia MFCC: np.ndarray [shape=(n_mfcc, t)]
                Secuencia transpuesta de MFCC de una canción generado por librosa.

            See Also
            --------
            librosa.feature.mfcc : Feature extraction
                (https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html)

        '''
        
        model_file = Path(model_path)
        if not model_file.exists():
            print("No se ha encontrado el fichero model json")
            sys.exit(0)

        with open(model_file) as f:
            config_model = f.read()

        self.model = model_from_json(config_model)
        self.model.summary()
        self.model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimizers.SGD(lr=0.001, momentum=0, decay=1e-5, nesterov=True),
                           metrics=['accuracy'])

    def loadWeights(self, weights_path):
        '''Carga los pesos al modelo.

            Parameters
            ----------
            weights_path : string
                Ruta del fichero que contiene los pesos del modelo.

        '''
        
        weights_file = Path(weights_path)
        if not weights_file.exists():
            print("No se ha encontrado el fichero con los pesos")
            sys.exit(0)

        self.model.load_weights(weights_path)

    def safeModel(self, log_dir):
        '''Guarda el modelo en un archivo JSON.

            Parameters
            ----------
            log_dir : string
                Ruta donde se guarda el modelo.

        '''        
        
        model_json = self.model.to_json()
        with open(log_dir, "w") as json_file:
            json_file.write(model_json)

    def safeWeights(self, log_dir):
        '''Guarda los pesos del modelo en un archivo h5py.

            Parameters
            ----------
            log_dir : string
                Ruta donde se guarda los pesos.

        '''        
        
        self.model.save_weights(log_dir + 'weights.hdf5')

    def trainModel(self, config, *data, callbacks):
        '''Entrena el modelo.

            Parameters
            ----------
            config : configparser
                Archivo con los parámetros de configuración.
            data: Arbitrary Argument Lists
                data[0] = X_train
                data[1] = y_train
                data[2] = X_test
                data[3] = y_test
                data[4] = X_val
                data[5] = y_val
            callbacks: Lista
                Lista de Callback de keras
                
            Returns
            -------
            history: History.history
                (https://keras.io/visualization/)

        ''' 

        history = self.model.fit(
                        data[0],
                        data[1],
                        batch_size=int(config['CNN_CONFIGURATION']['BATCH_SIZE']),
                        epochs=int(config['CNN_CONFIGURATION']['NUMBERS_EPOCH']),
                        verbose=1,
                        validation_data=(data[4], data[5]),
                        callbacks=callbacks)
        
        score = self.model.evaluate(data[2], data[3], verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        return history

    def predictModel(self, X_test):
        '''Genera predicciones de salida para las muestras de entrada.

            Parameters
            ----------
            X_test : np array
                Muestras de entrada
                
            Returns
            -------
            y_pred: np array
                Array con las predicciones

        '''

        Y_pred = self.model.predict(X_test)
        y_pred = np.argmax(Y_pred, axis=1)

        print("Predicho: ", y_pred)

        return y_pred


class CNNModel(Model):
    def buildModel(self, model_path, input_model, nb_classes):
        '''Crea un modelo a partir de los parámetros dados por un archivo json.

            Parameters
            ----------
            model_path : string
                Ruta del archivo json.
            input_model: Array
                Input de la red
            nb_classes: Entero
                Numero de clases a usar en la red
                
            See also
            -------
            filters: Tamaño del filtro
            kernel_size: Tamaño del Kernel para la convolución
            padding: Same or Valid
            pool_size: Tamaño para el area del Max Pooling

        ''' 

        model_path = Path(model_path)
        if not model_path.exists():
            print("No se ha encontrado el fichero model")
            sys.exit(0)

        with open(model_path) as json_data:
            model_json = json.load(json_data)

        model.add(Conv2D(
                        int(model_json['layer1']['filters']),
                        tuple(model_json['layer1']['kernel_size']),
                        padding=model_json['layer1']['padding'],
                        input_shape=(input_model.shape[1], input_model.shape[2], input_model.shape[3])))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=tuple(model_json['layer1']['pool_size'])))

        model.add(Conv2D(
                        int(model_json['layer2']['filters']),
                        tuple(model_json['layer2']['kernel_size']),
                        padding=model_json['layer2']['padding']))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=tuple(model_json['layer2']['pool_size'])))
        model.add(Dropout(float(model_json['layer2']['dropout'])))

        model.add(Conv2D(
                        int(model_json['layer3']['filters']),
                        tuple(model_json['layer3']['kernel_size']),
                        padding=model_json['layer3']['padding']))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=tuple(model_json['layer3']['pool_size'])))
        model.add(Dropout(float(model_json['layer3']['dropout'])))

        model.add(Conv2D(
                        int(model_json['layer4']['filters']),
                        tuple(model_json['layer4']['kernel_size']),
                        padding=model_json['layer4']['padding']))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=tuple(model_json['layer4']['pool_size'])))
        model.add(Dropout(float(model_json['layer4']['dropout'])))

        model.add(Flatten())

        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

        model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimizers.SGD(lr=0.001, momentum=0, decay=1e-5, nesterov=True),
                           metrics=['accuracy'])

        model.summary()
        
        
class LSTMModel(Model):
    def buildModel(self, model_path, input_model, nb_classes):
        """
            Creamos un modelo a partir de unos parámetros dados por un archivo json.
            # Arguments:
                model_path: String
                    Ruta del archivo json
                input_model: Array
                    Input de la red
                nb_classes: Entero
                    Numero de clases a usar en la red
            # Detalles:
        """
    
        model_path = Path(model_path)
        if not model_path.exists():
            print("No se ha encontrado el fichero model")
            sys.exit(0)
    
        with open(model_path) as json_data:
            model_json = json.load(json_data)
            
        self.model.add(LSTM(units=64, return_sequences=True, input_shape=(input_model.shape[1], input_model.shape[2])))
        self.model.add(LSTM(units=32, return_sequences=False))
    
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=10, activation='softmax'))
    
        self.model.compile(loss=losses.categorical_crossentropy,
                        optimizer=optimizers.RMSprop(lr=0.001),
                        metrics=['accuracy'])
    
        self.model.summary()