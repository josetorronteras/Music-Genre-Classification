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
        
        
    def build_model(self, model, nb_classes):
        """
            Crea el modelo en función de los parámetros establecidos.
            # Return:
                model: keras.model
        """
        model = Sequential()
        model.add(
                Conv2D(
                    int(model['layer1']['filters']),
                    tuple(model['layer1']['kernel_size']),
                    padding = model['layer1']['padding'],
                    input_shape = self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = tuple(model['layer1']['pool_size'])))
        
        model.add(
                Conv2D(
                    int(model['layer2']['filters']),
                    tuple(model['layer2']['kernel_size']),
                    padding = model['layer2']['padding']))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = tuple(model['layer2']['pool_size'])))
        model.add(Dropout(float(model['layer2']['dropout'])))

        model.add(
                Conv2D(
                    int(model['layer3']['filters']),
                    tuple(model['layer3']['kernel_size']),
                    padding = model['layer3']['padding']))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = tuple(model['layer3']['pool_size'])))
        model.add(Dropout(float(model['layer3']['dropout'])))
        
        model.add(
                Conv2D(
                    int(model['layer4']['filters']),
                    tuple(model['layer4']['kernel_size']),
                    padding = model['layer4']['padding']))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = tuple(model['layer4']['pool_size'])))
        model.add(Dropout(float(model['layer4']['dropout'])))
             
        model.add(Flatten())

        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.8))

        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))
        
        return model