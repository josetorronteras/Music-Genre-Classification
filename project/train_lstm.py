import configparser
import argparse
import sys
from pathlib import Path
from datetime import datetime

from source.get_train_test_data import GetTrainTestData
from source.lstm_model import LSTMModel
from source.aux_functions import plot_results_to_img

from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", help="Archivo de Configuración", required=True)
parser.add_argument("--device", "-v", type=int, default=0, help="Cuda Visible Device")
args = parser.parse_args()

config_path = Path(args.config)
if not config_path.exists():
    print("No se ha encontrado el fichero config")
    sys.exit(0)
config = configparser.ConfigParser()
config.read(config_path)

now = datetime.now()
model_id = 'lstm_' + now.isoformat() + '.json'

X_train, X_test, X_val, y_train, y_test, y_val = GetTrainTestData(config).read_dataset(choice="mfcc")

# Convertimos las clases a un array binario de clases
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_val = np_utils.to_categorical(y_val)

# Creamos los callbacks para el modelo
callbacks = [
    TensorBoard(log_dir=config['CALLBACKS']['TENSORBOARD_LOGDIR'] + model_id,
                write_images=config['CALLBACKS']['TENSORBOARD_WRITEIMAGES'],
                write_graph=config['CALLBACKS']['TENSORBOARD_WRITEGRAPH']
                ),
    EarlyStopping(monitor=config['CALLBACKS']['EARLYSTOPPING_MONITOR'],
                  mode=config['CALLBACKS']['EARLYSTOPPING_MODE'],
                  patience=int(config['LSTM_CONFIGURATION']['EARLYSTOPPING_PATIENCE']),
                  verbose=1)
]

model = LSTMModel(config)
model.generate_model((X_train.shape[1], X_train.shape[2]), y_test.shape[1])

# Entrenamos el modelo
history = model.train_model(callbacks, X_train, y_train, X_test, y_test, X_val, y_val)

# Grafica Accuracy
plot_results_to_img(
    model_id,
    config['CALLBACKS']['TENSORBOARD_LOGDIR'],
    'Model accuracy',
    (history.history['acc'], history.history['val_acc']),
    ('epoch', 'accuracy'))

# Grafica Loss
plot_results_to_img(
    model_id,
    config['CALLBACKS']['TENSORBOARD_LOGDIR'],
    'Model loss',
    (history.history['loss'], history.history['val_loss']),
    ('epoch', 'loss'))

# Guardamos el modelo
model.safe_model_to_file('./logs/' + model_id)
model.safe_weights_to_file('./logs/' + model_id)
