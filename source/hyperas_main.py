import os
os.environ["CUDA_VISIBLE_DEVICES"]="0";

from keras import optimizers
from keras import losses
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


import configparser
config = configparser.ConfigParser()
config.read('source/config-gpu.ini')

from Get_Train_Test_Data import GetTrainTestData

def data():
    X_train, X_test, X_val, y_train, y_test, y_val = GetTrainTestData(config).read_dataset()
 
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1).astype('float32')
    
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    y_val = np_utils.to_categorical(y_val)
    
    return X_train, X_test, X_val, y_train, y_test, y_val


def model(X_train, Y_train, X_test, Y_test):

    # Filtros a usar
    filters = 32
    # Tamaños para el area del Max Pooling
    pool_size = (2, 4) 
    # Tamaño Kernel para la Convolución
    kernel_size = (3, 3)
    input_shape = (X.shape[1], X.shape[2], X.shape[3])

    model = Sequential()
    model.add(
            Conv2D(
                32,
                kernel_size,
                padding='same',
                input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Activation('relu'))

    model.add(
            Conv2D(
                {{choice([64, 128, 256])}},
                kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(
            Conv2D(
                {{choice([128, 256, 512])}},
                kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(
            Conv2D(
                {{choice([256, 512, 1024])}},
                kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout({{uniform(0, 1)}}))
            
    model.add(Flatten())

    model.add(Dense({{choice([512, 1024])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(loss = losses.categorical_crossentropy,
                  #optimizer = optimizers.Adam(lr = 0.001),
                  optimizer = optimizers.SGD(lr = 0.001, momentum = 0, decay = 1e-5, nesterov = True),
                  metrics = ['accuracy'])
    
    model.summary()

    # Creamos los Callbacks
    callbacks = [
                ModelCheckpoint(filepath = config['CALLBACKS']['CHECKPOINT_FILE'],
                                verbose = 1,
                                save_best_only = True,
                            ),
                TensorBoard(log_dir = config['CALLBACKS']['TENSORBOARD_LOGDIR'],
                            write_images = config['CALLBACKS']['TENSORBOARD_WRITEIMAGES'],
                            write_graph = config['CALLBACKS']['TENSORBOARD_WRITEGRAPH'],
                            update_freq = config['CALLBACKS']['TENSORBOARD_UPDATEFREQ']
                            ),
                EarlyStopping(monitor = config['CALLBACKS']['EARLYSTOPPING_MONITOR'],
                            mode = config['CALLBACKS']['EARLYSTOPPING_MODE'], 
                            patience = int(config['CALLBACKS']['EARLYSTOPPING_PATIENCE']),
                            verbose = 1)
    ]

    # Entrenamos el modelo
    model.fit(
            X_train,
            y_train,
            batch_size ={ {choice([32, 64])}},,
            epochs = int(config['CNN_CONFIGURATION']['NUMBERS_EPOCH']),
            verbose = 1,
            validation_data = (X_val, y_val),
            callbacks = callbacks)

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':

    X_train, X_test, X_val, y_train, y_test, y_val = data()

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_val, y_val))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
