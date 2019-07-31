.. Guia Sphinx documentation master file, created by
   sphinx-quickstart on Thu May 11 20:34:32 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Clasificación de los géneros musicales usando deep learning
=======================================

.. toctree::
   :maxdepth: 3
   :caption: Contenidos:

   introduccion
   instalacion
   extracción de características
   creación del dataset
   modelos neuronales
   funciones auxiliares

Introducción
==================

* :ref:`genindex`
* :ref:`modindex`


Esta es la documentación del trabajo de fin de grado "Clasificación de los Géneros musicales
usando redes neuronales profundas", realizado por Jose Jesús Torronteras Hernández, alumno de 
Ingeniería Informática de la Universidad de Córdoba.
En este manual encontraremos el código de la aplicación ordenado por clases.
Por último es importante recordar que este manual es tan solo una referencia del código de la aplicación y que por tanto, para comprender 
completamente el funcionamiento de la misma, es imprescindible la consulta del Manual Técnico del Trabajo Fin de Grado.
El objetivo de este Trabajo ha sido la implementación de un modelo de red neuronal convolucional y recurrente que permita realizar una clasificación de géneros musicales haciendo uso de
la base de datos GTZAN. Además se ha implementado una interfaz de línea de comandos que permite entrenar y evaluar el modelo, además de realizar predicciones y preparar los datos de entrada.
Se ha utilizado el lenguaje de programación Python para este propósito, ya que es en un potente lenguaje orientado a objetos, y tiene una gran flexibilidad por el uso de memoria
dinámica, la extensión de nuevos módulos o la inclusión de cualquier tipo de aplicación.
En este manual explicaremos cada una de las clases y expondremos tanto la organización como la estructura de las mismos, poniendo  énfasis en las características más relevantes del 
diseño que se ha seguido, así como su funcionalidad y cometido dentro del conjunto del Trabajo Fin de Grado.


La tabla de contenidos muestra los diferentes apartados a tener en cuenta para entender tanto 
el código, como para ejecutar el proyecto de manera local.

Hay que tener en cuenta las siguiente
- Para crear un proyecto desde cero, ir a :doc:`Recetas <recetas/inicio>`.
- Para editar una documentación ya existente, ir a :doc:`Cómo editar
  <editando>`.
- Para ver algunos ejemplos de documentaciones ya existentes, ir a
  :doc:`Ejemplos <ejemplos>`.
- Como introducción, se puede ir a la página :doc:`Introducción <introduccion>`
  o ver estas :download:`diapositivas <../../presentacion.pdf>`.

Como esta guía está en *GitHub* y está hecha en *Sphinx*, se puede `descargar,
mirar, mandar correcciones y esas cosas`__.

__ Repositorio_

Notas
-----

* Cuando probé *Sphinx* en *Ubuntu* tuve un problema, al hacer ``make html``
  recibía un error que decía ``No module named sphinx``. Lo solucioné
  modificando el ``Makefile``, cambiando ``python -msphinx`` por
  ``sphinx-build``. Supongo que es algo relacionado a cambiar de *Python 2* a
  *Python 3*.

.. _glosario:

Glosario
--------

Las cosas que se usan para hacer esto son:

* **Python**: Es el lenguaje de programación que usamos para hacer el programa.
  Hay comentarios especiales que documentan una función, clase, etc. que se
  llaman **docstrings**.

* **Sphinx**: Es un programa que nos ayuda a generar la documentación para ese
  programa. Toma varios archivos escritos con **reStructuredText** y junto con
  los **docstrings** genera una página web estática.

* **reStructuredText**: Es un lenguaje de marcado, especifica como crear
  títulos, listas, tablas, cómo insertar imagenes, etc.

* **GitHub**: Es un sitio web que hostea sobre todo proyectos de software libre
  de forma gratuita. Permite hostear una web estática para cada proyecto, que
  viene perfecto para la documentación generada con **Sphinx** pero es opcional.

Esto es sobre *Python*:

* **módulo** o **script**: Es un archivo ``.py``.

* **paquete**: Es una carpeta que contiene archivos ``.py``. Debe tener un
  archivo (que puede estar vacío) llamado ``__init__.py``.

.. _Sphinx: http://www.sphinx-doc.org/en/stable/
.. _Repositorio: https://github.com/martinber/guia-sphinx
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _GitHub: https://github.com/