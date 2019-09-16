import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_results_to_img(id, log_dir, title, data, labels):
    """
    Genera las gráficas de los valores obtenidos durante el entrenamiento.
    Se guardan en formato '.png'

    :type id: string
    :type log_dir: string
    :type title: string
    :type data: tuple
    :type labels: tuple
    :param id: "Identificador de la ejecución"
    :param log_dir: "Ruta donde se guardará la gráfica"
    :param title: "Título de la gráfica"
    :param data: "Datos de entrada para la gráfica"
    :param labels: "Etiquetas de la gráfica"
    :return:
    """
    plt.title(title)
    plt.plot(data[0])
    plt.plot(data[1])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(log_dir + id + '.png')
    plt.close()


def create_confusion_matrix(y_test, y_pred):
    """
    Genera la visualización de una matriz de confusión
    Se guarda en formato '.png'
    :type y_test: numpy.array
    :type y_pred: numpy.array
    :param y_test: "Ids de las clases correctas"
    :param y_pred: "Ids de las clases predichas "
    :return:
    """
    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot()
    sns.heatmap(matrix, annot=True, ax=ax)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Matriz de confusión')
    ax.xaxis.set_ticklabels(['blues', 'classical', 'country', 'disco',
                             'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])
    ax.yaxis.set_ticklabels(['blues', 'classical', 'country', 'disco',
                             'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])

    plt.show()
