
Manual de Código
=================

Introducción
---------------

En el presente Manual de Código del Trabajo de Fin de Grado Clasificación de los géneros musicales mediante el uso de redes neuronales profundas, se expondrán de una manera detallada
todos los elementos que componen el sistema desarrollado por el mismo. El objetivo del presente documento es servir de manual de código, permitiendo así futuras implementaciones que
puedan desarollarse sobre la aplicación software.

En este manual encontraremos el código ordenado por clases.
Por último es importante recordad que este manual es tan solo una referencia del codifo de la aplicacion y que por tanto, para comprender completamente el funcionamiento de la misma,
es indispensable la consulta del Manual Técnico del Trabajo de Fin de Grado.

El objetivo de este trabajo de Fin de Grado ha sido la implementación de modelos de redes neuronales profundas, en concreto 

Características de la implementación
-------------------------------------

Notación de la implementación
*****************************
Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur
Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur

Estructura de la documentación
*******************************
Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur
Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur

Documentación del código
------------------------------

Clase ExtractAudioFeatures
**************************
En este apartado se especifica de forma concisa la clase Preprocess, que contiene los métodos que se han usado para preprocesar las canciones del conjunto de datos de GTZAN.Esta especificación se muestra en la Tabla XX.

.. automodule:: source.extract_audio_features
    :members:
    :undoc-members:
    :show-inheritance:
.. literalinclude:: ../project/source/extract_audio_features.py
    :language: python
    :linenos:

Clase GetTrainTestData
**************************
En este apartado se especifica de forma concisa la clase Dataset, encargada de la creacción de los datos previos al entrenamiento de los modelos.

.. automodule:: source.get_train_test_data
    :members:
    :undoc-members:
    :show-inheritance:
.. literalinclude:: ../project/source/get_train_test_data.py
    :language: python
    :linenos:

Clase CNNModel
**************************
En este apartado se especifica de forma concisa la clase CNN_Model, que contiene los métodos que permiten la creacción y posterior ejecucción de un modelo convolucional sobre los datos.

.. automodule:: source.cnn_model
    :members:
    :undoc-members:
    :show-inheritance:
.. literalinclude:: ../project/source/cnn_model.py
    :language: python
    :linenos:

Clase LSTMModel
**************************
En este apartado se especifica de forma concisa la clase LSTM_Model, que contiene los métodos que permiten la creacción y posterior ejecucción de un modelo recurrente sobre los datos.

.. automodule:: source.lstm_model
    :members:
    :undoc-members:
    :show-inheritance:
.. literalinclude:: ../project/source/lstm_model.py
    :language: python
    :linenos:

Funciones Auxiliares
**************************
En este apartado se especifica de forma concisa las funciones auxiliares implementadas.

.. automodule:: source.aux_functions
    :members:
    :undoc-members:
    :show-inheritance:
.. literalinclude:: ../project/source/aux_functions.py
    :language: python
    :linenos: