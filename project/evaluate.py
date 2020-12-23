import numpy as np
import argparse
from pathlib import Path
import sys
import configparser

from source.get_train_test_data import GetTrainTestData
from source.cnn_model import CNNModel
from source.lstm_model import LSTMModel
from source.aux_functions import create_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", type=str, help="Dataset a usar")
parser.add_argument("--logs", "-l", type=str, help="Directorio con los archivos del modelo entrenado")
parser.add_argument("--config", "-c", type=str, help="Archivo de Configuración", required=True)
args = parser.parse_args()

config_path = Path(args.config)
if not config_path.exists():
    print("No se ha encontrado el fichero config")
    sys.exit(0)
config = configparser.ConfigParser()
config.read(config_path)

X_train, X_test, X_val, y_train, y_test, y_val = GetTrainTestData(config).read_dataset(choice=args.dataset)

if args.dataset == "mfcc":
    model = LSTMModel(config)
    model.load_model(args.logs + 'model.json')
    model.load_weights(args.logs + 'weights.hdf5')
    y_pred = model.predict_model(X_test)
    create_confusion_matrix(y_test, y_pred)
elif args.dataset == "spec":
    model = CNNModel(config)
    model.load_model(args.logs + 'model.json')
    model.load_weights(args.logs + 'weights.hdf5')
    y_pred = model.predict_model(X_test)
    create_confusion_matrix(y_test, y_pred)
else:
    print("Error. Opción --dataset No válida.")