import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_results_to_img(model_id, log_dir, title, data, labels):
    """
    Genera las gráficas de los valores obtenidos durante el
    entrenamiento.
    Se guardan en formato '.png'.

    :type model_id: string
    :type log_dir: string
    :type title: string
    :type data: (np.array, np.array)
    :type labels: (string, string)
    :param model_id: "Identificador de la ejecución"
    :param log_dir: "Ruta donde se guardará la gráfica"
    :param title: "Título de la gráfica"
    :param data: "Datos de entrada para la gráfica"
    :param labels: "Etiquetas de la gráfica"
    """
    plt.title(title)
    plt.plot(data[0])
    plt.plot(data[1])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(log_dir + model_id + '/' + title + '.png')
    plt.close()


def create_confusion_matrix(y_test, y_pred):
    """
    Genera la visualización de una matriz de confusión.
    Se guarda en formato '.png'.

    :type y_test: numpy.array
    :type y_pred: numpy.array
    :param y_test: "Ids de las clases correctas"
    :param y_pred: "Ids de las clases predichas "
    """
    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot()
    sns.heatmap(matrix, annot=True, ax=ax)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Matriz de confusión')
    ax.xaxis.set_ticklabels(['blues', 'classical',
                             'country', 'disco',
                             'hiphop', 'jazz', 'metal',
                             'pop', 'reggae', 'rock'])
    ax.yaxis.set_ticklabels(['blues', 'classical',
                             'country', 'disco',
                             'hiphop', 'jazz',
                             'metal', 'pop',
                             'reggae', 'rock'])

    plt.show()


def get_name_dataset(config, choice):
    """
    Devuelve el nombre del dataset en función de los parámetros introducidos

    :type config: ConfigParser
    :type choice: string
    :param config: "Archivo con las configuraciones"
    :param choice: "Opción elegida"
    :rtype string
    :return Nombre del dataset
    """
    if choice == "spec":
        return config.get('DATA_CONFIGURATION', 'DATASET_PREPROCESSED_SPECTOGRAM')
    elif choice == "mfcc":
        return config.get('DATA_CONFIGURATION', 'DATASET_PREPROCESSED_MFCC')
