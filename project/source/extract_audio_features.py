import os
from pathlib import Path
import librosa
import numpy as np
import h5py
import threading
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler

from source.aux_functions import get_name_dataset

class ExtractAudioFeatures(object):

    def __init__(self, config):
        """
        :type config: ConfigParser
        :param config: "Contiene las rutas de los archivos de
        audio y los parámetros a usar."
        """

        self.config = config

        # Rutas de los ficheros
        self.dest = config['PATH_CONFIGURATION']['AUDIO_PATH']
        self.path = config['PATH_CONFIGURATION']['DATASET_PATH']

        # Parámetros Librosa
        self.n_mels = int(config['AUDIO_FEATURES']['N_MELS'])
        self.n_fft = int(config['AUDIO_FEATURES']['N_FFT'])
        self.n_mfcc = int(config['AUDIO_FEATURES']['N_MFCC'])
        self.hop_length = int(config['AUDIO_FEATURES']['HOP_LENGTH'])
        self.duration = int(config['AUDIO_FEATURES']['DURATION'])

        self.options = {
            "spec": [
                "Generar Espectograma Mel", self.get_melspectogram
            ],
            "mfcc": [
                "Generar Coeficientes Espectrales Mel", self.get_spectral_features
            ]
        }

        self.lck = threading.Lock()

    def get_melspectogram(self, file_path):
        """
        Genera el Espectograma Mel de una canción

        :type file_path: string
        :param file_path: "Ruta de una canción"
        :rtype: np.array
        :return: Mel spectrogram [shape=(n_mels, t)]
        :also: librosa.feature.melspectrogram : Feature extraction
        """
        # Cargamos el audio con librosa
        y, sr = librosa.load(file_path, duration=self.duration)
        s = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y,
                sr=sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length),
            ref=np.max)
        # Transforms features by scaling each feature to a given range.
        s = MinMaxScaler().fit_transform(s.reshape(-1, s.shape[1])).reshape(s.shape[0], s.shape[1])
        s = s.reshape(s.shape[0], s.shape[1], 1).astype('float32')
        return s

    def get_spectral_features(self, file_path):
        """
        Extrae los Mel Frequency Cepstral Coeficientes de una
        canción
        Son coeﬁcientes para la representación del habla basados en
        la percepción auditiva humana

        :type file_path: string
        :param file_path: "Ruta de una canción"
        :rtype: np.array
        :return: Secuencia MFCC [shape=(n_mfcc, t)]
                Secuencia transpuesta de MFCC de una canción
                generado por librosa.
        :also: librosa.feature.mfcc : Feature extraction
        """
        # Cargamos el audio con librosa
        y, sr = librosa.load(file_path, duration=self.duration)
        mfcc = librosa.feature.mfcc(y=y,
                                    sr=sr,
                                    hop_length=self.hop_length,
                                    n_mfcc=self.n_mfcc)
        return mfcc.T

    def runner(self, directorio, dataset_name, action):
        """
          Ejecución principal del hilo.

          :type directorio: string
          :type dataset_name: string
          :type action: callable
          :param directorio: "Contiene el nombre del directorio
          donde se encuentran los audios de un género."
          :param dataset_name: "Nombre del dataset"
          :param action: "Método de preprocesamiento elegido"
        """
        group_hdf_dict = {}
        for root, subdirs, files in os.walk(self.path + directorio + '/'):
            for filename in tqdm(files):
                # Descartamos ficheros .DS_store
                if filename.endswith('.wav'):
                    file_path = Path(root, filename)
                    try:
                        s = action(file_path)
                        group_hdf_dict[filename] = s
                    except Exception as e:
                        print("Error accured" + str(e))

        self.lck.acquire()
        with h5py.File(dataset_name, 'a') as hdf:
            group_hdf = hdf.create_group(str(directorio))
            for key, value in group_hdf_dict.items():
                group_hdf.create_dataset(key,
                                         data=value,
                                         compression='gzip')
            # Limpiamos memoria
            del directorio, dataset_name
            del group_hdf_dict
        self.lck.release()

    def prepossessing_audio(self, choice):
        """
        Preprocesamiento del Dataset GTZAN, para la creacción del Dataset
        Crea un archivo h5py con todos los datos generados

        :type choice: string
        :param choice: "Método de preprocesamiento elegido"
        """
        check_option = self.options.get(choice)
        if check_option is not None:
            action = check_option[1]
        else:
            print("Error. Opción --prepocess No válida.")
            print("Seleccione spec o mfcc")
            raise SystemExit

        # Obtenemos una lista de los directorios
        directorios = [nombre_directorio
                       for nombre_directorio
                       in os.listdir(self.path)
                       if os.path.isdir(os.path.join(self.path, nombre_directorio))]
        directorios.sort()

        # Obtenemos el nombre del dataset
        dataset_name = get_name_dataset(self.config, choice)

        threads = []
        for i in range(0, 10):
            t = threading.Thread(target=self.runner,
                                 args=(directorios[i], self.dest + dataset_name, action))
            threads.append(t)
            t.start()
