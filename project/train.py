import configparser
import argparse
import sys
from pathlib import Path

from source.get_train_test_data import GetTrainTestData
from source.cnn_model import CNNModel
from source.aux_functions import pltResults

from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", help="Archivo de Configuración", required=True)
args = parser.parse_args()

config_path = Path(args.config)
if not config_path.exists():
    print("No se ha encontrado el fichero config")
    sys.exit(0)
config = configparser.ConfigParser()
config.read(config_path)

X_train, X_test, \
X_val, y_train, \
y_test, y_val = GetTrainTestData(config).read_dataset(choice=args.dataset)

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

# Creamos los callbacks para el modelo
callbacks = [
    TensorBoard(log_dir=config['CALLBACKS']['TENSORBOARD_LOGDIR'] + "CNN1",
                write_images=config['CALLBACKS']['TENSORBOARD_WRITEIMAGES'],
                write_graph=config['CALLBACKS']['TENSORBOARD_WRITEGRAPH'],
                update_freq=config['CALLBACKS']['TENSORBOARD_UPDATEFREQ']
                ),
    EarlyStopping(monitor=config['CALLBACKS']['EARLYSTOPPING_MONITOR'],
                  mode=config['CALLBACKS']['EARLYSTOPPING_MODE'],
                  patience=int(config['CALLBACKS']['EARLYSTOPPING_PATIENCE']),
                  verbose=1)
]

model = CNNModel()
model.generate_model((X_train.shape[1], X_train.shape[2], X_train.shape[3]), y_test.shape[1])

# Entrenamos el modelo
history = model.train_model(config, callbacks, X_train,
                           y_train, X_test,
                           y_test, X_val,
                           y_val)

# Grafica Accuracy
pltResults(
    config['CALLBACKS']['TENSORBOARD_LOGDIR'] + "CNN1",
    history.history['acc'],
    history.history['val_acc'],
    'Model accuracy',
    'epoch',
    'accuracy')

# Grafica Loss
pltResults(
    config['CALLBACKS']['TENSORBOARD_LOGDIR'] + "CNN1",
    history.history['loss'],
    history.history['val_loss'],
    'Model loss',
    'epoch',
    'loss')

# Guardamos el modelo
model.safeModel('./logs/model.json')
model.safeWeights('./logs/')