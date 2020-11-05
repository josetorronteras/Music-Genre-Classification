import sys
import numpy as np
from pathlib import Path
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers.recurrent import LSTM
from keras import optimizers, losses


class LSTMModel:

    def __init__(self, config):
        """
        Inicializador de la clase 

        :type config: ConfigParser
        :param config: "Contiene las rutas de los archivos de
        audio y los parámetros a usar."
        """
        self.model = Sequential()
        self.batch_size = int(config['LSTM_CONFIGURATION']['BATCH_SIZE'])
        self.epochs = int(config['LSTM_CONFIGURATION']['NUMBERS_EPOCH'])
        self.learning_rate = float(config['LSTM_CONFIGURATION']['LEARNING_RATE'])

    def train_model(self, callbacks, *data):
        """
        Entrena el modelo

        :type callbacks: list
        :type data: array
        :param callbacks: "Conjunto de funciones que se ejecutarán
        durante el entrenamiento"
        :param data: array "Dataset a entrenar"
        :rtype: History.history
        :return: "Registro de los valores obtenidos durante el entrenamiento"
        """
        history = self.model.fit(
            data[0],
            data[1],
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=(data[4], data[5]),
            callbacks=callbacks)

        score = self.model.evaluate(data[2], data[3], verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        
        return history

    def load_model(self, model_path):
        """
        Carga un fichero de keras

        :type model_path: string
        :param model_path: "Ruta del fichero"
        """
        model_file = Path(model_path)
        if not model_file.exists():
            print("No se ha encontrado el fichero model json")
            sys.exit(0)

        with open(model_file) as f:
            config_model = f.read()

        self.model = model_from_json(config_model)
        self.model.summary()
        self.model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimizers.RMSprop(lr=self.learning_rate),
                           metrics=['accuracy'])

    def load_weights(self, weights_path):
        """
        Cargamos los pesos en el modelo.
        
        :type weights_path: string
        :param weights_path: "Ruta del fichero que contiene los pesos del modelo."
        """
        weights_file = Path(weights_path)
        if not weights_file.exists():
            print("No se ha encontrado el fichero con los pesos")
            sys.exit(0)
        self.model.load_weights(weights_path)

    def safe_model_to_file(self, log_dir):
        """
        Guarda el modelo en un archivo Json

        :type log_dir: string
        :param log_dir: "Ruta donde se guarda el modelo"
        """
        model_json = self.model.to_json()
        with open(log_dir + '/model.json', "w") as json_file:
            json_file.write(model_json)

    def safe_weights_to_file(self, log_dir):
        """
        Guarda los pesos del modelo en un archivo h5py

        :type log_dir: string
        :param log_dir: "Ruta donde se guardan los pesos"
        """
        self.model.save_weights(log_dir + '/weights.hdf5')

    def generate_model(self, input_model, number_classes):
        """
        Genera el modelo LSTM

        :type input_model: tuple
        :type number_classes: int
        :param input_model: tuple "Shape del conjunto de entrada
        de la red"
        :param number_classes: int "Número de clases del dataset"
        """
        self.model.add(LSTM(units=64,
                            return_sequences=True,
                            input_shape=input_model))
        self.model.add(LSTM(units=32,
                            return_sequences=False))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(number_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimizers.RMSprop(lr=self.learning_rate),
                           metrics=['accuracy'])
        self.model.summary()

    def predict_model(self, X_test):
        """
        Realiza la predicción sobre los valores introducidos.
        :type  X_test: Array
        :param X_test: "Numpy array con los datos a predecir."
        """
        Y_pred = self.model.predict(X_test)
        y_pred = np.argmax(Y_pred, axis=1)

        print("Predicho: ", y_pred)
        return y_pred