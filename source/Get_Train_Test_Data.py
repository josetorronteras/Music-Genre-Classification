import numpy as np
from tqdm import tqdm
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class GetTrainTestData(object):
    """
        Prepara y divide los datos preprocesados para un posterior entrenamiento.
        # Arguments:
            object: configparser
                Archivo con las distintas configuraciones.
    """


    def __init__(self, config):
        # Rutas de los ficheros
        self.DATASET_PATH   = config['PATH_CONFIGURATION']['AUDIO_PATH']

        # Nombre del dataset generado
        self.DATASET_NAME   = config['DATA_CONFIGURATION']['DATASET_NAME']

        # Configuración de los datos
        self.SIZE           = int(config['DATA_CONFIGURATION']['DATA_SIZE'])
        self.SPLIT_SIZE     = float(config['DATA_CONFIGURATION']['SPLIT_SIZE'])


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
        
        print("Getting.." + self.DATASET_PATH + genre) # ???????????????????????
        
        # Leemos los datos
        for items in tqdm(dataset_file[genre]):
            # Comprobamos el límite de lectura
            if limit == self.SIZE:
                break
            # Introducimos los datos
            else:
                read_data.append(dataset_file[genre][items][()])
                limit +=1

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
        dataset_file = h5py.File(self.DATASET_PATH + self.DATASET_NAME, 'r')
        
        # Obtenemos los arrays de cada género
        arr_blues       = self.getDataFromDataset('blues', dataset_file)
        arr_classical   = self.getDataFromDataset('classical', dataset_file)
        arr_country     = self.getDataFromDataset('country', dataset_file)
        arr_disco       = self.getDataFromDataset('disco', dataset_file)
        arr_hiphop      = self.getDataFromDataset('hiphop', dataset_file)
        arr_jazz        = self.getDataFromDataset('jazz', dataset_file)
        arr_metal       = self.getDataFromDataset('metal', dataset_file)
        arr_pop         = self.getDataFromDataset('pop', dataset_file)
        arr_reggae      = self.getDataFromDataset('reggae', dataset_file)
        arr_rock        = self.getDataFromDataset('rock', dataset_file)

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

        # Transforms features by scaling each feature to a given range.
        # features = MinMaxScaler().fit_transform(full_data.reshape(-1, full_data.shape[2])).reshape(full_data.shape[0], full_data.shape[1], full_data.shape[2])

        # With train_test_split() it is more easier obtain the necessary elements for the later learning.
        print("test-size = " + str(self.SPLIT_SIZE) + " Change value in config.py") # We can change the size in the config file.
        print("data-size = " + str(self.SIZE) + " Change value in config.py") # We can change the size in the config file.

        # Dividimos los datos, en función a SPLIT_SIZE (config)
        X_train, X_test, y_train, y_test = train_test_split(
                                                            full_data,
                                                            labels,
                                                            test_size = self.SPLIT_SIZE,
                                                            random_state = 0,
                                                            stratify = labels)
        
        X_test, X_val, y_test, y_val = train_test_split(
                                                        X_test,
                                                        y_test,
                                                        test_size = 0.5,
                                                        random_state = 0,
                                                        stratify = y_test)

        # Guardamos los datos generados
        with h5py.File(self.DATASET_PATH + 'traintest.hdf5', 'w') as hdf:
            hdf.create_dataset('X_train', data = X_train, compression = 'gzip')
            hdf.create_dataset('y_train', data = y_train, compression = 'gzip')
            hdf.create_dataset('X_test', data = X_test, compression = 'gzip')
            hdf.create_dataset('y_test', data = y_test, compression = 'gzip')
            hdf.create_dataset('X_val', data = X_val, compression = 'gzip')
            hdf.create_dataset('y_val', data = y_val, compression = 'gzip')

        print("X_train Tamaño: %s - X_test Tamaño: %s - X_val Tamaño: %s - y_train Tamaño: %s - y_test Tamaño: %s - y_val Tamaño: %s " % (X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape))

    
    def read_dataset(self):
        """
            Lee el dataset dividido.
            # Return:
                dataset: np array
                    X_train y_train X_test y_test X_val y_val
        """
        dataset = h5py.File(self.DATASET_PATH + 'traintest.hdf5', 'r')
        return dataset['X_train'][()], dataset['X_test'][()], dataset['X_val'][()], dataset['y_train'][()], dataset['y_test'][()], dataset['y_val'][()]