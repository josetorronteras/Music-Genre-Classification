import sys
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class GetTrainTestData(object):
    """
        Prepara y divide los datos preprocesados para un posterior entrenamiento.
        # Arguments:
            object: configparser
                Archivo con las distintas configuraciones.
    """

    def __init__(self, config):
        # Rutas de los ficheros
        self.DATASET_PATH = config['PATH_CONFIGURATION']['AUDIO_PATH']

        # Nombre del dataset generado
        self.DATASET_NAME = config['DATA_CONFIGURATION']['DATASET_NAME']

        # Configuración de los datos
        self.SIZE = int(config['DATA_CONFIGURATION']['DATA_SIZE'])
        self.SPLIT_SIZE = float(config['DATA_CONFIGURATION']['SPLIT_SIZE'])

    def getDataFromDataset(self, genre, dataset_file):
        """
            Recoge el array de espectogramas del género deseado del dataset preprocesado.
            # Arguments:
                genre: string
                    Nombre del Género del que se desea obtener los arrays.
                dataset_file: h5py File
                    Dataset File que contiene los datos preprocesados.
            # Return:
                read_data: list(np.array)
                    Lista que contiene todos los arrays del género seleccionado.
        """
        # Lista que acumula los datos leidos del conjunto de datos.
        read_data = []

        # Establece un límite de lectura
        limit = 0

        print("Obteniendo.." + self.DATASET_PATH + genre)

        # Leemos los datos
        for items in tqdm(dataset_file[genre]):
            # Comprobamos el límite de lectura
            if limit == self.SIZE:
                break
            # Introducimos los datos
            # Escalamos los datos entre 0 y 1
            else:
                read_data.append((dataset_file[genre][items][()]))
                limit += 1

        # features_arr = np.vstack(aux_list)
        return read_data

    def splitDataset(self):
        """
            Divide el dataset en X_train X_test X_val para el entrenamiento.
            Se guardan en un fichero h5py.
            # Example:
                ```
                    python main.py --dataset --config=CONFIGFILE.ini
                ```
        """
        dataset_file_path = Path(self.DATASET_PATH + self.DATASET_NAME)
        if not dataset_file_path.exists():
            print("No se ha encontrado el fichero")
            sys.exit(0)

        dataset_file = h5py.File(dataset_file_path, 'r')
        
        # Obtenemos los arrays de cada género
        arr_blues = self.getDataFromDataset('blues', dataset_file)
        arr_classical = self.getDataFromDataset('classical', dataset_file)
        arr_country = self.getDataFromDataset('country', dataset_file)
        arr_disco = self.getDataFromDataset('disco', dataset_file)
        arr_hiphop = self.getDataFromDataset('hiphop', dataset_file)
        arr_jazz = self.getDataFromDataset('jazz', dataset_file)
        arr_metal = self.getDataFromDataset('metal', dataset_file)
        arr_pop = self.getDataFromDataset('pop', dataset_file)
        arr_reggae = self.getDataFromDataset('reggae', dataset_file)
        arr_rock = self.getDataFromDataset('rock', dataset_file)

        # Los agrupamos
        full_data = np.vstack((arr_blues,\
                            arr_classical,\
                            arr_country,\
                            arr_disco,\
                            arr_hiphop,\
                            arr_jazz,\
                            arr_metal,\
                            arr_pop,\
                            arr_reggae,\
                            arr_rock))

        # Establecemos las etiquetas que identifican el género musical
        labels = np.concatenate((np.zeros(len(arr_blues)),\
                                np.ones(len(arr_classical)),\
                                np.full(len(arr_country), 2),\
                                np.full(len(arr_disco), 3),\
                                np.full(len(arr_hiphop), 4),\
                                np.full(len(arr_jazz), 5),\
                                np.full(len(arr_metal), 6),\
                                np.full(len(arr_pop), 7),\
                                np.full(len(arr_reggae), 8),\
                                np.full(len(arr_rock), 9)))

        # Con train_test_split() dividimos los datos.
        print("test-size = " + str(self.SPLIT_SIZE) + " Cambiar valor en config.py") # Se puede cambiar el tamaño en el archivo config.
        print("data-size = " + str(self.SIZE) + " Cambiar valor en config.py") # Se puede cambiar el tamaño en el archivo config.

        # Dividimos los datos, en función a SPLIT_SIZE (config)
        X_train, X_test, y_train, y_test = train_test_split(
                                                    full_data,
                                                    labels,
                                                    test_size=self.SPLIT_SIZE,
                                                    stratify=labels)
        
        X_test, X_val, y_test, y_val = train_test_split(
                                                X_test,
                                                y_test,
                                                test_size=0.5,
                                                stratify=y_test)

        # Guardamos los datos generados
        dataset_output_path = Path(self.DATASET_PATH + 'traintest.hdf5')
        with h5py.File(dataset_output_path, 'w') as hdf:
            hdf.create_dataset('X_train', data=X_train, compression='gzip')
            hdf.create_dataset('y_train', data=y_train, compression='gzip')
            hdf.create_dataset('X_test', data=X_test, compression='gzip')
            hdf.create_dataset('y_test', data=y_test, compression='gzip')
            hdf.create_dataset('X_val', data=X_val, compression='gzip')
            hdf.create_dataset('y_val', data=y_val, compression='gzip')

        print("X_train Tamaño: %s - X_test Tamaño: %s - X_val Tamaño: %s - y_train Tamaño: %s - y_test Tamaño: %s - y_val Tamaño: %s " % \
             (X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape))
    
    def read_dataset(self):
        """
            Lee el dataset dividido.
            # Return:
                dataset: np array
                    X_train y_train X_test y_test X_val y_val
        """
        dataset_file_path = Path(self.DATASET_PATH + 'traintest.hdf5')
        if not dataset_file_path.exists():
            print("No se ha encontrado el fichero")
            sys.exit(0)

        dataset = h5py.File(dataset_file_path, 'r')
        return dataset['X_train'][()], dataset['X_test'][()],\
                dataset['X_val'][()], dataset['y_train'][()],\
                dataset['y_test'][()], dataset['y_val'][()]
