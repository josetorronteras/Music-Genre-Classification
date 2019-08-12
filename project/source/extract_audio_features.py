"""
Extracción de las características de cada canción
"""
import os
from pathlib import Path
import librosa
import numpy as np
import h5py
import threading

from tqdm import tqdm


class ExtractAudioFeatures(object):
    """
        Clase ExtractAudioFeatures.

        Parameters
        ----------
        object: Argumentos para la clase
            Contiene las rutas de los archivos de audio y los parámetros a usar.

    """

    def __init__(self, config):
        """Inicializador.

            Parameters
            ----------
            config: Fichero configparser
                Contiene las rutas de los archivos de audio y los parámetros a usar.
                :type config: ConfigParser

        """

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

        self.options = {
            "spec": ["Generar Espectograma Mel", self.getMelspectogram],
            "mfcc": ["Generar Coeficientes Espectrales Mel", self.spectralFeatures]
        }

        self.lck = threading.Lock()

    def getMelspectogram(self, file_Path):
        """Genera el Espectograma Mel de una canción.

            Parameters
            ----------
            file_Path : string
            :param file_Path:
                Ruta de la canción

            Returns
            -------
            :return: Mel spectrogram: np.array [shape=(n_mels, t)]
                Espectograma escalado de una canción generado por librosa.

            See Also
            --------
            librosa.feature.melspectrogram : Feature extraction
                (https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html)



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

        return S

    def spectralFeatures(self, file_path):
        """Extraer los Mel Frequency Cepstral Coeficientes de una canción.

            Son coeﬁcientes para la representación del habla basados en la percepción auditiva humana.

            Parameters
            ----------
            :param file_path: string
                Ruta de la canción

            Returns
            -------
            Secuencia MFCC: np.ndarray [shape=(n_mfcc, t)]
                Secuencia transpuesta de MFCC de una canción generado por librosa.

            See Also
            --------
            librosa.feature.mfcc : Feature extraction
                (https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html)
        """

        # Cargamos el audio con librosa
        y, sr = librosa.load(file_path, duration=self.DURATION)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=self.HOP_LENGTH, n_mfcc=13)

        return mfcc.T

    def runner(self, directorio, elegir_nombre_dataset, action):
        """Ejecución del Hilo.

            Parameters
            ----------
            directorio: String
                Contiene el nombre del directorio donde se encuentran los audios de un género.
                :type directorio: String
            elegir_nombre_dataset: String
            :param elegir_nombre_dataset: String
                Contiene el nombre del dataset
        """
        group_hdf_dict = {}
        for root, subdirs, files in os.walk(self.PATH + directorio + '/'):
            for filename in tqdm(files):
                if filename.endswith('.au'):  # Descartamos otros ficheros .DS_store
                    file_path = Path(root, filename)  # Ruta de la cancion
                    try:
                        S = action(file_path)
                        group_hdf_dict[filename] = S
                    except Exception as e:
                        print("Error accured" + str(e))

        self.lck.acquire()
        with h5py.File(elegir_nombre_dataset, 'a') as hdf:
            group_hdf = hdf.create_group(str(directorio))
            for key, value in group_hdf_dict.items():
                group_hdf.create_dataset(
                    key,
                    data=value,
                    compression='gzip')  # Incluimos el fichero numpy en el dataset.
            # Limpiamos memoria
            del directorio, elegir_nombre_dataset
            del files, root, subdirs
            del S, group_hdf_dict
        self.lck.release()

    def prepossessingAudio(self, choice):
        """Preprocesamiento del Dataset GTZAN, para la creacción del Dataset.

            Crea un archivo h5py con todos los datos generados.

            Parameters
            ----------
            choice : string
                Método de procesamiento elegido

            Examples
            --------
            python main.py --preprocess=["spec" or "mfcc"] --config=CONFIGFILE.ini
        """

        check_option = self.options.get(choice)
        if check_option is not None:
            action = check_option[1]
        else:
            print("Error. Opción --prepocess No válida.")
            print("Seleccione spec o mfcc")
            raise SystemExit

        # Obtenemos una lista de los directorios
        directorios = [nombre_directorio for nombre_directorio in os.listdir(self.PATH)
                       if os.path.isdir(os.path.join(self.PATH, nombre_directorio))]
        directorios.sort()

        # Cambiamos el nombre del dataset en función de lo deseado
        elegir_nombre_dataset = lambda nombre: Path(self.DEST + self.DATASET_NAME_SPECTOGRAM) if choice == "spec" \
            else Path(self.DEST + self.DATASET_NAME_MFCC)

        threads = []
        for i in range(0, 10):
            t = threading.Thread(target = self.runner, args = (directorios[i], elegir_nombre_dataset(choice), action))
            threads.append(t)
            t.start()