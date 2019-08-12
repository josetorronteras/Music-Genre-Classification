import configparser
import argparse
import sys
from pathlib import Path

from source.extract_audio_features import ExtractAudioFeatures

parser = argparse.ArgumentParser()
parser.add_argument("--preprocess", "-p", type=str, help="Preprocesar las canciones.")
parser.add_argument("--config", "-c", help="Archivo de Configuración", required=True)
args = parser.parse_args()

config_path = Path(args.config)
if not config_path.exists():
    print("No se ha encontrado el fichero config")
    sys.exit(0)
config = configparser.ConfigParser()
config.read(config_path)

ExtractAudioFeatures(config).prepossessingAudio(choice=args.preprocess)
