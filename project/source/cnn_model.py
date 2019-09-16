import sys
from pathlib import Path
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import optimizers, losses


class CNNModel:

    def __init__(self):
        self.model = Sequential()

    def train_model(self, config, callbacks, *data):
        """
        :type config: ConfigParser
        :type callbacks: list
        :type data: array
        :param config: "Fichero con los parámetros de configuración"
        :param callbacks: "Conjunto de funciones que se ejecutarán durante el entrenamiento"
        :param data: array "Dataset a entrenar"
        :rtype: History.history
        :return: "Registro de los valores obtenidos durante el entrenamiento"
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

    def load_model(self, model_path):
        """
        Carga un fichero de keras
        :type model_path: string
        :param model_path: "Ruta del fichero"
        :return:
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

    def safe_model_to_file(self, log_dir):
        """
        Guarda el modelo en un archivo Json
        :type log_dir: string
        :param log_dir: "Ruta donde se guarda el modelo"
        :return:
        """
        model_json = self.model.to_json()
        with open(log_dir, "w") as json_file:
            json_file.write(model_json)

    def safe_weights_to_file(self, log_dir):
        """
        Guarda los pesos del modelo en un archivo h5py
        :type log_dir: string
        :param log_dir: "Ruta donde se guardan los pesos"
        :return:
        """
        self.model.save_weights(log_dir + 'weights.hdf5')
        
    def generate_model(self, input_model, number_classes):
        """
        :type input_model: tuple
        :type number_classes: int
        :param input_model: tuple "Shape del conjunto de entrada de la red"
        :param number_classes: int "Número de clases del dataset"
        :return:
        """

        # Conv1
        self.model.add(Conv2D(32, (11, 11), padding="same", input_shape=input_model))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv2
        self.model.add(Conv2D(64, (11, 11), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 4)))
        self.model.add(Dropout(0.25))

        # Conv3
        self.model.add(Conv2D(128, (11, 11), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(4, 4)))
        self.model.add(Dropout(0.25))

        # Conv4
        self.model.add(Conv2D(256, (11, 11), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(4, 4)))
        self.model.add(Dropout(0.25))

        # FC
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(number_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimizers.SGD(lr=0.001, momentum=0, decay=1e-5, nesterov=True),
                           metrics=['accuracy'])

        self.model.summary()