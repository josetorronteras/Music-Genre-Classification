import configparser
import argparse
import sys
from pathlib import Path
from source.get_train_test_data import GetTrainTestData

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", type=str, help="Preparar los datos para el entrenamiento")
parser.add_argument("--config", "-c", help="Archivo de Configuración", required=True)
args = parser.parse_args()

config_path = Path(args.config)
if not config_path.exists():
    print("No se ha encontrado el fichero config")
    sys.exit(0)
config = configparser.ConfigParser()
config.read(config_path)

GetTrainTestData(config).splitDataset(choice=args.dataset)
