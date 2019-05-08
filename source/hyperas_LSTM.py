import os
import argparse
import numpy as np
from pathlib import Path
import json

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", help = "Archivo de Configuracion", required = True)
parser.add_argument("--device", "-v", type = int, default = 0, help = "Cuda Visible Device")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device);

import configparser

from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras import optimizers
from keras import losses
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe

from Get_Train_Test_Data import GetTrainTestData
from LSTM_Model import LSTMModel

config = configparser.ConfigParser()
config.read('config/config-gpu.ini')


def data():
    config = configparser.ConfigParser()
    config.read('config/config-gpu.ini')

    X_train, X_test, X_val, y_train, y_test, y_val = GetTrainTestData(config).read_dataset(spectogram=False)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    y_val = np_utils.to_categorical(y_val)

    return X_train, X_test, X_val, y_train, y_test, y_val

def hyperas_model(X_train, y_train, X_test, y_test, X_val, y_val):
    
    config = configparser.ConfigParser()
    config.read('config/config-gpu.ini')

    # Creamos los callbacks para el modelo
    callbacks = [
                EarlyStopping(monitor=config['CALLBACKS']['EARLYSTOPPING_MONITOR'],
                            mode=config['CALLBACKS']['EARLYSTOPPING_MODE'], 
                            patience=int(config['CALLBACKS']['EARLYSTOPPING_PATIENCE']),
                            verbose=1)
    ]

    model = Sequential()
    model.add(LSTM(units={{choice([32, 64, 128, 256, 512])}}, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Activation('relu'))
    if {{choice(['una', 'dos'])}} == 'dos':
        model.add(LSTM(units={{choice([32, 64, 128])}}, return_sequences=False))
        model.add(Activation('relu'))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss=losses.categorical_crossentropy,
                    optimizer={{choice(['adam', 'sgd'])}},
                    metrics=['accuracy'])

    model.summary()

    result = model.fit(
                    X_train,
                    y_train,
                    batch_size=int(config['CNN_CONFIGURATION']['BATCH_SIZE']),
                    epochs=int(config['CNN_CONFIGURATION']['NUMBERS_EPOCH']),
                    verbose=1,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks)
                    
    validation_acc = np.amax(result.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}



best_run, best_model = optim.minimize(model = hyperas_model,
                                        data = data,
                                        algo = tpe.suggest,
                                        max_evals = 50,
                                        trials = Trials())

X_train, X_test, X_val, y_train, y_test, y_val = data()

print("Evalutation of best performing model:")
print(best_model.evaluate(X_val, y_val))
print("Best performing model chosen hyper-parameters:")
print(best_run)
json.dump(best_run, open("best_run.txt", 'w'))