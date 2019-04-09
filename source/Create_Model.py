from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


class CNNModel(object):
    """
        Creacción del modelo CNN
        # Arguments:
            object: configparser
                Archivo con las distintas configuraciones.
    """


    def __init__(self, config, X):

        # Filtros a usar
        self.filters = 32
        # Tamaños para el area del Max Pooling
        self.pool_size = (2, 4) 
        # Tamaño Kernel para la Convolución
        self.kernel_size = (3, 3)
        self.input_shape = (X.shape[1], X.shape[2], X.shape[3])
        
        
    def build_model(self, nb_classes):
        """
            Crea el modelo en función de los parámetros establecidos.
            # Return:
                model: keras.model
        """
        model = Sequential()
        model.add(
                Conv2D(
                    self.filters,
                    self.kernel_size,
                    padding='same',
                    input_shape = self.input_shape))
        model.add(MaxPooling2D(pool_size = self.pool_size))
        model.add(Activation('relu'))
        
        model.add(
                Conv2D(
                    128,
                    self.kernel_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = self.pool_size))
        model.add(Dropout(0.1))

        model.add(
                Conv2D(
                    128,
                    self.kernel_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = self.pool_size))
        model.add(Dropout(0.1))
        
        model.add(
                Conv2D(
                    1024,
                    self.kernel_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = self.pool_size))
        model.add(Dropout(0.25))
             
        model.add(Flatten())

        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.8))

        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))
        
        return model