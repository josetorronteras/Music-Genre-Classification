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
from keras import optimizers
from keras import losses


class CNNModel():
    """
        Creacción del modelo CNN
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
        """
            Cargamos los pesos en el modelo.
            # Arguments:
                weights_path: String
                    Ruta del fichero que contiene los pesos del modelo.
        """
        weights_file = Path(weights_path)
        if not weights_file.exists():
            print("No se ha encontrado el fichero con los pesos")
            sys.exit(0)

        self.model.load_weights(weights_path)

    def safeModel(self, log_dir):
        """
            Guardamos el modelo en un archivo json.
            # Arguments:
                log_dir: String
                    Ruta donde se guarda el modelo.
        """
        model_json = self.model.to_json()
        with open(log_dir, "w") as json_file:
            json_file.write(model_json)

    def safeWeights(self, log_dir):
        """
            Guardamos los pesos del modelo en un archivo h5py.
            # Arguments:
                log_dir: String
                    Ruta donde se guarda los pesos.
        """
        self.model.save_weights(log_dir + 'weights.hdf5')

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
                filters: Tamaño del filtro
                kernel_size: Tamaño del Kernel para la convolución
                padding: Same o Valid
                pool_size: Tamaño para el area del Max Pooling
        """

        model_path = Path(model_path)
        if not model_path.exists():
            print("No se ha encontrado el fichero model")
            sys.exit(0)

        with open(model_path) as json_data:
            model_json = json.load(json_data)

        self.model.add(Conv2D(
                        int(model_json['layer1']['filters']),
                        tuple(model_json['layer1']['kernel_size']),
                        padding=model_json['layer1']['padding'],
                        input_shape=(input_model.shape[1], input_model.shape[2], input_model.shape[3])))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=tuple(model_json['layer1']['pool_size'])))

        self.model.add(Conv2D(
                        int(model_json['layer2']['filters']),
                        tuple(model_json['layer2']['kernel_size']),
                        padding=model_json['layer2']['padding']))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=tuple(model_json['layer2']['pool_size'])))
        self.model.add(Dropout(float(model_json['layer2']['dropout'])))

        self.model.add(Conv2D(
                        int(model_json['layer3']['filters']),
                        tuple(model_json['layer3']['kernel_size']),
                        padding=model_json['layer3']['padding']))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=tuple(model_json['layer3']['pool_size'])))
        self.model.add(Dropout(float(model_json['layer3']['dropout'])))

        self.model.add(Conv2D(
                        int(model_json['layer4']['filters']),
                        tuple(model_json['layer4']['kernel_size']),
                        padding=model_json['layer4']['padding']))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=tuple(model_json['layer4']['pool_size'])))
        self.model.add(Dropout(float(model_json['layer4']['dropout'])))

        self.model.add(Flatten())

        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(nb_classes))
        self.model.add(Activation("softmax"))

        self.model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimizers.SGD(lr=0.001, momentum=0, decay=1e-5, nesterov=True),
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
        """
            Guardamos los pesos del modelo en un archivo h5py.
            # Arguments:
                X_test: Array
                    Numpy array con los datos ha predecir.
        """
        Y_pred = self.model.predict(X_test)
        y_pred = np.argmax(Y_pred, axis=1)

        print("Predecido: ", y_pred)

        return y_pred
