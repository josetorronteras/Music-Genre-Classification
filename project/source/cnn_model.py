from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers, losses

class CNNModel:

    def __init__(self):
        self.model = Sequential()

    def train_model(self, config, callbacks, *data):

        history = self.model.fit(
            data[0],
            data[1],
            batch_size=int(config['CNN_CONFIGURATION']['BATCH_SIZE']),
            epochs=int(config['CNN_CONFIGURATION']['NUMBERS_EPOCH']),
            verbose=1,
            validation_data=(data[4], data[5]),
            callbacks=callbacks)

        score = self.model.evaluate(data[2], data[3], verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        return history

    def generate_model(self, input_model, number_classes):

        # Conv1
        self.model.add(Conv2D(32, (11, 11), padding="same", input_shape=input_model))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.25))

        # Conv2
        self.model.add(Conv2D(128, (11, 11), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 4)))
        self.model.add(Dropout(0.25))

        # Conv3
        self.model.add(Conv2D(128, (11, 11), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(4, 4)))
        self.model.add(Dropout(0.25))

        # Conv4
        self.model.add(Conv2D(127, (11, 11), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(4, 4)))
        self.model.add(Dropout(0.25))

        # FC
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(number_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimizers.SGD(lr=0.001, momentum=0, decay=1e-5, nesterov=True),
                           metrics=['accuracy'])

        self.model.summary()
