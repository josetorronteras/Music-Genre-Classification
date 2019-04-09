import configparser
import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0";

from keras import optimizers
from keras import losses
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
from os.path import isfile

from Extract_Audio_Features import ExtractAudioFeatures
from Get_Train_Test_Data import GetTrainTestData
from Create_Model import CNNModel

parser = argparse.ArgumentParser()
parser.add_argument("--preprocess", "-p", help = "Preparar los datos de las canciones", action = "store_true")
parser.add_argument("--dataset", "-d", help = "Preparar los datos para el entrenamiento", action = "store_true")
parser.add_argument("--trainmodel", "-t", help = "Entrenar el modelo", action = "store_true")
parser.add_argument("--config", "-c", help = "Archivo de Configuracion")
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config)

if args.preprocess:
    ExtractAudioFeatures(config).prepossessingAudio()
elif args.dataset:
    GetTrainTestData(config).splitDataset()
elif args.trainmodel:
    # Leemos los datos
    X_train, X_test, X_val, y_train, y_test, y_val = GetTrainTestData(config).read_dataset()
 
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1).astype('float32')
    
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    y_val = np_utils.to_categorical(y_val)

    # Creamos el modelo
    model = CNNModel(config, X_train).build_model(nb_classes = y_test.shape[1])

    model.compile(loss = losses.categorical_crossentropy,
                  #optimizer = optimizers.Adam(lr = 0.001),
                  optimizer = optimizers.SGD(lr = 0.001, momentum = 0, decay = 1e-5, nesterov = True),
                  metrics = ['accuracy'])
    model.summary()
    
    # Guardamos el Modelo
    model_json = model.to_json()
    with open(config['CALLBACKS']['TENSORBOARD_LOGDIR'] + "model.json", "w") as json_file:
        json_file.write(model_json)

    # Comprobamos si hay un fichero checkpoint
    if int(config['CALLBACKS']['LOAD_CHECKPOINT']):
        print("Buscando fichero Checkpoint...")
        if isfile(config['CALLBACKS']['CHECKPOINT_FILE']):
            print('Fichero Checkpoint detectando. Cargando weights.')
            model.load_weights(config['CALLBACKS']['CHECKPOINT_FILE'])
        else:
            print('No se ha detectado el fichero Checkpoint.  Empezando de cero')
    else:
        print('No Checkpoint')
    
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
            batch_size = int(config['CNN_CONFIGURATION']['BATCH_SIZE']),
            epochs = int(config['CNN_CONFIGURATION']['NUMBERS_EPOCH']),
            verbose = 1,
            validation_data = (X_val, y_val),
            callbacks = callbacks)

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model.save_weights(config['PATH_CONFIGURATION']['OUTPUT'] + config['OUTPUT']['WEIGHTS_FILE'])