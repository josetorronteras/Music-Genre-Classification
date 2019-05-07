import os
from pathlib import Path
from sklearn import preprocessing
import librosa
import numpy as np
import h5py

from tqdm import tqdm

class ExtractAudioFeatures(object):
    """
        Genera los espectogramas de cada canción haciendo uso de librosa.
        # Arguments:
            object: configparser
                Archivo con las distintas configuraciones.
    """

    def __init__(self, config):
        # Rutas de los ficheros
        self.DEST = config['PATH_CONFIGURATION']['AUDIO_PATH']
        self.PATH = config['PATH_CONFIGURATION']['DATASET_PATH']

        # Nombre del dataset generado
        self.DATASET_NAME_SPECTOGRAM = config['DATA_CONFIGURATION']['DATASET_NAME_SPECTOGRAM']
        self.DATASET_NAME_MFCC = config['DATA_CONFIGURATION']['DATASET_NAME_MFCC']

        # Parámetros Librosa
        self.N_MELS = int(config['AUDIO_FEATURES']['N_MELS'])
        self.N_FFT = int(config['AUDIO_FEATURES']['N_FFT'])
        self.HOP_LENGTH = int(config['AUDIO_FEATURES']['HOP_LENGTH'])
        self.DURATION = int(config['AUDIO_FEATURES']['DURATION'])

    def getMelspectogram(self, file_Path):
        """
            Calcula el espectograma de una canción y lo transforma a dB
            para una representación gráfica.
            # Arguments:
                file_Path: string
                    Ruta del fichero de audio.
            # Return:
                S: np.array
                    Imagen de un Espectograma en dB. Dividimos entre 80 para tenerlo escalado.
        """
        # Cargamos el audio con librosa
        y, sr = librosa.load(file_Path, duration=self.DURATION)

        S = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y,
                sr=sr,
                n_mels=self.N_MELS,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH),
            ref=np.max)

        return S/80

    def spectralFeatures(self, file_path):
        """

        """
        # Cargamos el audio con librosa
        y, sr = librosa.load(file_path, duration=self.DURATION)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=self.HOP_LENGTH)
        #spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.HOP_LENGTH)
        #chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.HOP_LENGTH)
        #spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.HOP_LENGTH)
        # np.vstack((mfcc, spectral_center, chroma, spectral_contrast))
        return preprocessing.scale(mfcc)

    def prepossessingAudio(self, spectogram = True):
        """
            Preprocesamiento de GTZAN, para la creacción del Dataset.
            Crea un archivo h5py con todos los datos generados.
            # Arguments:
                spectogram: Bool
                    Genera el dataset del espectograma
            # Example:
                ```
                    python main.py --preprocess --config=CONFIGFILE.ini
                ```
        """
        # Obtenemos una lista de los directorios
        directorios = [nombre_directorio for nombre_directorio in os.listdir(self.PATH) \
                        if os.path.isdir(os.path.join(self.PATH, nombre_directorio))]
        directorios.sort()
        directorios.insert(0, directorios[0])

        # Cambiamos el nombre del dataset en función de lo deseado
        elegirNombreDataset = lambda spectogram: Path(self.DEST + self.DATASET_NAME_SPECTOGRAM) if spectogram \
                                else Path(self.DEST + self.DATASET_NAME_MFCC)
        # Escribimos el Dataset Preprocesado en formato h5py
        with h5py.File(elegirNombreDataset(spectogram), 'w') as hdf:

            for root, subdirs, files in os.walk(self.PATH):
                # Ordenamos las carpetas por orden alfabético
                subdirs.sort()

                try:
                    # Creamos un nuevo grupo con el nombre del directorio en el que estamos.
                    group_hdf = hdf.create_group(directorios[0]) 
                except Exception as e:
                    print("Error accured " + str(e))

                for filename in tqdm(files):
                    if filename.endswith('.au'): # Descartamos otros ficheros .DS_store
                        file_Path = Path(root, filename) # Ruta de la cancion
                        print('Fichero %s (full path: %s)' % (filename, file_Path))

                        try:
                            if spectogram:
                                # Obtenemos las caracteristicas espectrales
                                S = self.getMelspectogram(file_Path)
                            else:
                                # Obtenemos el Mel-Spectogram
                                S = self.spectralFeatures(file_Path)

                            group_hdf.create_dataset(
                                filename,
                                data=S,
                                compression='gzip') # Incluimos el fichero numpy en el dataset.
                        except Exception as e:
                            print("Error accured" + str(e))
                directorios.pop(0) # Next directorio
            # Limpiamos memoria
            del directorios, file_Path
            del files, root, subdirs
            del S
