import sys
import json
from pathlib import Path
import numpy as np

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras import optimizers
from keras import losses

class LSTMModel():
    """
        Creacción del modelo LSTM
    """

    def __init__(self):
        self.model = Sequential()

    def loadModel(self, model_path):
        """
            Lee un archivo con la configuración del modelo.
            # Arguments:
                model_path: String
                    Ruta del fichero que contiene la información del modelo.
        """
        print("keep this function tmp")

    def loadWeights(self, weights_path):
        """
            Cargamos los pesos en el modelo.
            # Arguments:
                weights_path: String
                    Ruta del fichero que contiene los pesos del modelo.
        """
        print("keep this function tmp")

    def safeModel(self, log_dir):
        """
            Guardamos el modelo en un archivo json.
            # Arguments:
                log_dir: String
                    Ruta donde se guarda el modelo.
        """
        print("keep this function tmp")

    def safeWeights(self, log_dir):
        """
            Guardamos los pesos del modelo en un archivo h5py.
            # Arguments:
                log_dir: String
                    Ruta donde se guarda los pesos.
        """
        print("keep this function tmp")

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
        
    def trainModel(self, config, *data, callbacks):
        """
            Entrenamos el modelo junto con los datos de entrada.
            # Arguments:
                config: configparser
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
        """
        history = self.model.fit(
                        data[0],
                        data[1],
                        batch_size=int(config['LSTM_CONFIGURATION']['BATCH_SIZE']),
                        epochs=int(config['LSTM_CONFIGURATION']['NUMBERS_EPOCH']),
                        verbose=1,
                        validation_data=(data[4], data[5]),
                        callbacks=callbacks)
        
        score = self.model.evaluate(data[2], data[3], verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        return history

    def predictModel(self, X_test):
        """
            Guardamos los pesos del modelo en un archivo h5py.
            # Arguments:
                X_test: Array
                    Numpy array con los datos ha predecir.
        """
        print("keep this function tmp")
        
