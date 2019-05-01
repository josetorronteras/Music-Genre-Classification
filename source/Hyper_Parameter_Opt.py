import os
import configparser
import argparse
import matplotlib.pyplot as plt
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.utils import use_named_args
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", help = "Archivo de Configuracion", required = True)
parser.add_argument("--device", "-v", type = int, default = 0, help = "Cuda Visible Device")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device);

from keras import optimizers
from keras import losses
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from Get_Train_Test_Data import GetTrainTestData

config_path = Path(args.config)
if not config_path.exists():
    print("No se ha encontrado el fichero config")
    sys.exit(0)
config = configparser.ConfigParser()
config.read(config_path)

def log_dir_name(learning_rate, dense, filters1, filters2, filters3, filters4, kernel, maxpool):
    # The dir-name for the TensorBoard log-dir.
    s = "./logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}_{4}_{5}_{6}_{7}/"
    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate, dense, filters1, filters2, filters3, filters4, kernel, maxpool)
    return log_dir

def createModel(learning_rate, dense, filters1, filters2, filters3, filters4, kernel, maxpool):
    model = Sequential()
    model.add(
            Conv2D(
                filters1,
                kernel,
                padding = "Same",
                input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = maxpool))
    
    model.add(
            Conv2D(
                filters2,
                kernel,
                padding = "Same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = maxpool))
    model.add(Dropout(0.25))

    model.add(
            Conv2D(
                filters3,
                kernel,
                padding = "Same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = maxpool))
    model.add(Dropout(0.25))
    
    model.add(
            Conv2D(
                filters4,
                kernel,
                padding = "Same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = maxpool))
    model.add(Dropout(0.25))
            
    model.add(Flatten())

    model.add(Dense(dense))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation("softmax"))
    
    model.compile(loss = losses.categorical_crossentropy,
            #optimizer = optimizers.Adam(lr = 0.001),
            optimizer = optimizers.SGD(lr = learning_rate),
            metrics = ['accuracy'])
    model.summary()

    return model

X_train, X_test, X_val, y_train, y_test, y_val = GetTrainTestData(config).read_dataset()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1).astype('float32')

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_val = np_utils.to_categorical(y_val)


dim_learning_rate = Real(low = 1e-4, high = 1e-2, prior = 'log-uniform', name = 'learning_rate')
dim_num_dense = Integer(low = 32, high = 1024, name = 'dense')
dim_num_filters_layer1 = Categorical( categories =[32, 64, 128], name = 'filters1')
dim_num_filters_layer2 = Categorical( categories = [64, 128, 256], name = 'filters2')
dim_num_filters_layer3 = Categorical( categories = [256, 512, 1024], name = 'filters3')
dim_num_filters_layer4 = Categorical( categories = [256, 512, 1024], name = 'filters4')
dim_num_kernel = Categorical( categories = [(3, 3), (4, 4), (6, 6)], name = "kernel")
dim_num_maxpool = Categorical( categories = [(2, 2), (2, 4), (4, 4)], name = "maxpool")

dimensions = [dim_learning_rate, dim_num_dense, dim_num_filters_layer1, dim_num_filters_layer2, dim_num_filters_layer3, dim_num_filters_layer4, dim_num_kernel, dim_num_maxpool]
default_parameters = [1e-3, 512, 32, 128, 128, 256, (3, 3), (2, 4)]

best_accuracy = 0.0
path_best_model = 'best_model.keras'

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, dense, filters1, filters2, filters3, filters4, kernel, maxpool):
    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('dense:', dense)
    print('filters1:', filters1)
    print('filters2:', filters2)
    print('filters3:', filters3)
    print('filters4:', filters4)
    print('kernel:', kernel)
    print('maxpool:', maxpool)
    print()
    
    accuracy = 0
    try:
        # Create the neural network with these hyper-parameters.
        model = createModel(learning_rate, dense, filters1, filters2, filters3, filters4, kernel, maxpool)
        # Dir-name for the TensorBoard log-files.
        log_dir = log_dir_name(learning_rate, dense, filters1, filters2, filters3, filters4, kernel, maxpool)
        
        # Create a callback-function for Keras which will be
        # run after each epoch has ended during training.
        # This saves the log-files for TensorBoard.
        # Note that there are complications when histogram_freq=1.
        # It might give strange errors and it also does not properly
        # support Keras data-generators for the validation-set.

        callbacks = [
                    TensorBoard(log_dir = log_dir,
                                write_images = config['CALLBACKS']['TENSORBOARD_WRITEIMAGES'],
                                write_graph = config['CALLBACKS']['TENSORBOARD_WRITEGRAPH'],
                                update_freq = config['CALLBACKS']['TENSORBOARD_UPDATEFREQ']
                                ),
                    EarlyStopping(monitor = config['CALLBACKS']['EARLYSTOPPING_MONITOR'],
                                mode = config['CALLBACKS']['EARLYSTOPPING_MODE'], 
                                patience = int(config['CALLBACKS']['EARLYSTOPPING_PATIENCE']),
                                verbose = 1)
        ]

        history = model.fit(
                            X_train,
                            y_train,
                            batch_size = int(config['CNN_CONFIGURATION']['BATCH_SIZE']),
                            epochs = int(config['CNN_CONFIGURATION']['NUMBERS_EPOCH']),
                            verbose = 1,
                            validation_data = (X_val, y_val),
                            callbacks = callbacks)
        # Get the classification accuracy on the validation-set
        # after the last training-epoch.
        accuracy = history.history['val_acc'][-1]

        # Print the classification accuracy.
        print()
        print("Accuracy: {0:.2%}".format(accuracy))
        print()

        # Save the model if it improves on the best-found performance.
        # We use the global keyword so we update the variable outside
        # of this function.
        global best_accuracy

        # Grafica Accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(log_dir +  '/acc.png')
        plt.close()

        # Grafica Loss 
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(log_dir + '/loss.png')
        plt.close()

        # If the classification accuracy of the saved model is improved ...
        if accuracy > best_accuracy:
            # Save the new model to harddisk.
            #model.save(path_best_model)
            # Update the classification accuracy.
            best_accuracy = accuracy

        # Delete the Keras model with these hyper-parameters from memory.
        del model
        
        # Clear the Keras session, otherwise it will keep adding new
        # models to the same TensorFlow graph each time we create
        # a model with a different set of hyper-parameters.
        K.clear_session()
        
    except:
        pass

    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -accuracy

search_result = gp_minimize(func = fitness,
                            dimensions = dimensions,
                            acq_func = 'EI', # Expected Improvement.
                            n_calls = 40)

plot_convergence(search_result)
plt.savefig(config['CALLBACKS']['TENSORBOARD_LOGDIR'] + '/opt.png')
plt.close()

space = search_result.space
space.point_to_dict(search_result.x)
print(sorted(zip(search_result.func_vals, search_result.x_iters)))