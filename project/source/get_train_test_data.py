import sys
from pathlib import Path
import h5py
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import os
from source.aux_functions import get_name_dataset

class GetTrainTestData(object):

    def __init__(self, config):
        """
        :type config: ConfigParser
        :param config: "Contiene las rutas de los archivos de audio
        y los parámetros a usar"
        """
        self.config = config

        # Rutas de los ficheros
        self.dataset_path = config['PATH_CONFIGURATION']['AUDIO_PATH']

        # Configuración de los datos
        self.size = int(config['DATA_CONFIGURATION']['DATA_SIZE'])
        self.splite_size = float(config['DATA_CONFIGURATION']['SPLIT_SIZE'])

        self.options = {
            "spec": ["Dataset Espectograma Mel"],
            "mfcc": ["Dataset Coeficientes Espectrales Mel"]
        }

    def get_data_from_dataset(self, genre, dataset_file):
        """
        Recoge las características extraidas de todas las canciones
        de un género establecido

        :type genre: string
        :type dataset_file: h5py file
        :param genre: "Nombre del Género del que se desea obtener
        los arrays"
        :param dataset_file: "Archivo h5py con los datos
        preprocesados"
        :rtype: np.array
        :return: read_data: "Lista que contiene los arrays del género seleccionado"
        """
        read_data = []
        print("Obteniendo.." + self.dataset_path + genre)

        for items in tqdm(dataset_file[genre]):
            read_data.append((dataset_file[genre][items][()]))
        return read_data

    def split_dataset(self, choice):
        """
        Divide el dataset para el entrenamiento y crea las
        etiquetas de los datos
        Se guarda en un fichero h5py

        :type choice: string
        :param choice: "Preprocesamiento elegido"
        """
        check_option = self.options.get(choice)
        if check_option is None:
            print("Error. Opción --dataset No válida.")
            print("Seleccione spec o mfcc")
            raise SystemExit

        # Obtenemos el nombre del dataset
        dataset_name = get_name_dataset(self.config, choice)

        if not Path(self.dataset_path + dataset_name).exists():
            print("No se ha encontrado el fichero" + self.dataset_path + dataset_name)
            sys.exit(0)

        dataset_file = h5py.File(Path(self.dataset_path + dataset_name), 'r')

        # Obtenemos los arrays de cada género
        arr_blues = self.get_data_from_dataset('blues', dataset_file)
        arr_classical = self.get_data_from_dataset('classical', dataset_file)
        arr_country = self.get_data_from_dataset('country', dataset_file)
        arr_disco = self.get_data_from_dataset('disco', dataset_file)
        arr_hiphop = self.get_data_from_dataset('hiphop', dataset_file)
        arr_jazz = self.get_data_from_dataset('jazz', dataset_file)
        arr_metal = self.get_data_from_dataset('metal', dataset_file)
        arr_pop = self.get_data_from_dataset('pop', dataset_file)
        arr_reggae = self.get_data_from_dataset('reggae', dataset_file)
        arr_rock = self.get_data_from_dataset('rock', dataset_file)

        # Los agrupamos
        full_data = np.vstack((arr_blues,
                               arr_classical,
                               arr_country,
                               arr_disco,
                               arr_hiphop,
                               arr_jazz,
                               arr_metal,
                               arr_pop,
                               arr_reggae,
                               arr_rock))

        # Establecemos las etiquetas que identifican el género musical
        labels = np.concatenate((np.zeros(len(arr_blues)),
                                 np.ones(len(arr_classical)),
                                 np.full(len(arr_country), 2),
                                 np.full(len(arr_disco), 3),
                                 np.full(len(arr_hiphop), 4),
                                 np.full(len(arr_jazz), 5),
                                 np.full(len(arr_metal), 6),
                                 np.full(len(arr_pop), 7),
                                 np.full(len(arr_reggae), 8),
                                 np.full(len(arr_rock), 9)))

        del arr_blues, arr_classical, arr_country, \
            arr_disco, arr_hiphop, arr_jazz, arr_metal, \
            arr_pop, arr_reggae, arr_rock

        # Con train_test_split() dividimos los datos.
        # Se puede cambiar el tamaño en el archivo config.
        print("test-size = " + str(self.splite_size))
        # Se puede cambiar el tamaño en el archivo config.
        print("data-size = " + str(self.size))

        # Dividimos los datos, en función a SPLIT_SIZE (config)
        X_train, X_test, y_train, y_test = train_test_split(
            full_data,
            labels,
            test_size=self.splite_size,
            stratify=labels)

        X_test, X_val, y_test, y_val = train_test_split(
            X_test,
            y_test,
            test_size=0.5,
            stratify=y_test)

        del full_data, labels

        # Guardamos los datos generados
        dataset_output_path = Path(self.dataset_path + choice + '/' + 'traintest_' + dataset_name)
        with h5py.File(dataset_output_path, 'w') as hdf:
            hdf.create_dataset('X_train',
                               data=X_train,
                               compression='gzip')
            hdf.create_dataset('y_train',
                               data=y_train,
                               compression='gzip')
            hdf.create_dataset('X_test',
                               data=X_test,
                               compression='gzip')
            hdf.create_dataset('y_test',
                               data=y_test,
                               compression='gzip')
            hdf.create_dataset('X_val',
                               data=X_val,
                               compression='gzip')
            hdf.create_dataset('y_val',
                               data=y_val,
                               compression='gzip')

        print("X_train Tamaño: %s - X_test Tamaño: %s - X_val Tamaño: %s\
              - y_train Tamaño: %s - y_test Tamaño: %s - y_val Tamaño: %s " %
              (X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape))

    def read_dataset(self, choice):
        """
        Lee el dataset seleccionado.

        :type choice: string
        :param choice: "Dataset preprocesado elegido"
        :rtype: (int, numpy.array(float))
        :return: Conjunto de datos para el entrenamiento
        """
        # Obtenemos el nombre del dataset
        dataset_name = get_name_dataset(self.config, choice)
        dataset_output_path = Path(self.dataset_path + choice + '/' + 'traintest_' + dataset_name)
        if not dataset_output_path.exists():
            print(dataset_output_path)
            print("No se ha encontrado el fichero")
            sys.exit(0)

        dataset = h5py.File(dataset_output_path, 'r')
        return dataset['X_train'][()], dataset['X_test'][()],\
               dataset['X_val'][()], dataset['y_train'][()],\
               dataset['y_test'][()], dataset['y_val'][()]
