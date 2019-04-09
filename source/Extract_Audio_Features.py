import librosa
import numpy as np
import h5py
import os
import sys

class ExtractAudioFeatures(object):
    """
        Genera los espectogramas de cada canción haciendo uso de librosa.
        # Arguments:
            object: configparser
                Archivo con las distintas configuraciones.
    """


    def __init__(self, config):
        # Rutas de los ficheros
        self.DEST           = config['PATH_CONFIGURATION']['AUDIO_PATH']
        self.PATH           = config['PATH_CONFIGURATION']['DATASET_PATH']

        # Nombre del dataset generado
        self.DATASET_NAME   = config['DATA_CONFIGURATION']['DATASET_NAME']

        # Parámetros Librosa
        self.N_MELS         = int(config['AUDIO_FEATURES']['N_MELS'])
        self.N_FFT          = int(config['AUDIO_FEATURES']['N_FFT'])
        self.HOP_LENGTH     = int(config['AUDIO_FEATURES']['HOP_LENGTH'])
        self.DURATION       = int(config['AUDIO_FEATURES']['DURATION'])
    
    
    def librosaAudio(self, file_Path):
        """
            Calcula el espectograma de una canción y lo transforma a dB
            para una representación gráfica.
            # Arguments: 
                file_Path: string
                    Ruta del fichero de audio.
            # Return:
                S: np.array
                    Imagen de un Espectograma en dB.
        """
        # Cargamos el audio con librosa
        y, sr = librosa.load(file_Path, duration = self.DURATION)

        S = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y,
                    sr = sr,
                    n_mels = self.N_MELS,
                    n_fft = self.N_FFT,
                    hop_length = self.HOP_LENGTH),
                    ref = np.max)
        
        return S


    def prepossessingAudio(self):
        """
            Preprocesamiento de GTZAN, para la creacción del Dataset.
            Crea un archivo h5py con todos los datos generados.
            # Example:
                ```
                    python main.py --preprocess --config=CONFIGFILE.ini
                ```
        """
        progress = 1.0
        
        # Obtenemos una lista de los directorios
        directorios = [nombre_directorio for nombre_directorio in os.listdir(self.PATH) if os.path.isdir(os.path.join(self.PATH, nombre_directorio))]
        directorios.sort()
        directorios.insert(0, directorios[0])
        
        # Escribimos el Dataset Preprocesado en formato h5py
        with h5py.File(self.DEST + self.DATASET_NAME, 'w') as hdf:

            for root, subdirs, files in os.walk(self.PATH):
                # Ordenamos las carpetas por orden alfabético
                subdirs.sort()

                try:
                    # Creamos un nuevo grupo con el nombre del directorio en el que estamos.
                    group_hdf = hdf.create_group(directorios[0]) 
                except Exception as e:
                    print("Error accured" + str(e))

                for filename in files:
                    if filename.endswith('.au'): # Descartamos otros ficheros .DS_store
                        file_Path = os.path.join(root, filename) # Ruta de la cancion
                        print('Fichero %s (full path: %s)' % (filename, file_Path))
                            
                        try:
                            S = self.librosaAudio(file_Path) # Obtenemos el Mel-Spectogram
                            group_hdf.create_dataset(filename, data = S, compression = 'gzip') # Incluimos el fichero numpy en el dataset.
                        except Exception as e:
                            print("Error accured" + str(e))

                        porcentaje = progress / 10
                        sys.stdout.write("\n%f%%  " % porcentaje)
                        sys.stdout.flush()
                        progress += 1
                directorios.pop(0) # Next directorio
            # Limpiamos memoria
            del directorios, porcentaje, progress, file_Path
            del files, root, subdirs
            del S