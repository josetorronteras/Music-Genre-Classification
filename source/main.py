import configparser
import argparse
import os
import sys
from pathlib import Path

from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping

from Extract_Audio_Features import ExtractAudioFeatures
from Get_Train_Test_Data import GetTrainTestData
from CNN_Model import CNNModel
from Aux_Functions import pltResults

parser = argparse.ArgumentParser()
parser.add_argument("--preprocess", "-p", help="Preparar los datos de las canciones", action="store_true")
parser.add_argument("--dataset", "-d", help="Preparar los datos para el entrenamiento", action="store_true")
parser.add_argument("--trainmodel", "-t", help="Entrenar el modelo", action="store_true")
parser.add_argument("--model", "-m ", help="Archivo con los parámetros del modelo")
parser.add_argument("--kerasmodel", "-k ", help="Archivo json con el modelo de keras")
parser.add_argument("--config", "-c", help="Archivo de Configuracion", required=True)
parser.add_argument("--device", "-v", type=int, default=0, help="Cuda Visible Device")
args = parser.parse_args()

# Seleccionamos la gpu disponible. Por defecto la 0.
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device);

config_path = Path(args.config)
if not config_path.exists():
    print("No se ha encontrado el fichero config")
    sys.exit(0)
config = configparser.ConfigParser()
config.read(config_path)

if args.preprocess:
    # Preprocesamos los datos
    ExtractAudioFeatures(config).prepossessingAudio()
elif args.dataset:
    # Creamos el dataset
    GetTrainTestData(config).splitDataset()
elif args.trainmodel:
    # Leemos el dataset
    X_train, X_test, X_val, y_train, y_test, y_val = GetTrainTestData(config).read_dataset()

    # Transformamos el shape de los datos
    X_train = X_train.reshape(
        X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
    X_test = X_test.reshape(
        X_test.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
    X_val = X_val.reshape(
        X_val.shape[0], X_val.shape[1], X_val.shape[2], 1).astype('float32')

    # Convertimos las clases a una matriz binaria de clases
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    y_val = np_utils.to_categorical(y_val)

    # Creamos el modelo
    model = CNNModel()

    # Cargamos el modelo
    if args.kerasmodel:
        model.loadModel(args.kerasmodel)
    elif args.model:
        model.buildModel(args.model, X_train, y_test.shape[1])

    # Creamos los callbacks para el modelo
    callbacks = [
                TensorBoard(log_dir=config['CALLBACKS']['TENSORBOARD_LOGDIR'],
                            write_images=config['CALLBACKS']['TENSORBOARD_WRITEIMAGES'],
                            write_graph=config['CALLBACKS']['TENSORBOARD_WRITEGRAPH'],
                            update_freq=config['CALLBACKS']['TENSORBOARD_UPDATEFREQ']
                            ),
                EarlyStopping(monitor=config['CALLBACKS']['EARLYSTOPPING_MONITOR'],
                            mode=config['CALLBACKS']['EARLYSTOPPING_MODE'], 
                            patience=int(config['CALLBACKS']['EARLYSTOPPING_PATIENCE']),
                            verbose=1)
    ]

    # Entrenamos el modelo
    history = model.trainModel(config, X_train, y_train, X_test, y_test, X_val, y_val, callbacks)

    # Grafica Accuracy
    pltResults(
        config, 
        history.history['acc'], 
        history.history['val_acc'], 
        'Model accuracy', 
        'epoch', 
        'accuracy')

    # Grafica Loss 
    pltResults(
        config, 
        history.history['loss'], 
        history.history['val_loss'], 
        'Model loss', 
        'epoch', 
        'loss')

    # Guardamos el modelo
    model.safeModel('./logs/model.json')
    model.safeWeights('./logs/')
    