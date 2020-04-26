# -*- coding: utf-8 -*-
"""
This module builds and compiles the Keras ANN model.


@author: kalivoda
"""

import keras
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, BatchNormalization, Dropout, AveragePooling2D, Flatten, Dense
from keras.models import Sequential


class ANN:

    def __init__(self, channels, interval):
        self.epochs = 30
        self.model = Sequential((
            Conv2D(6, (3, 3),
                   activation='elu',
                   input_shape=(channels, interval, 1)
                   ),
            BatchNormalization(),
            Dropout(0.5),
            AveragePooling2D(pool_size=(1, 8)),
            Flatten(),
            Dense(100, activation='elu'),
            BatchNormalization,
            Dropout(0.5),
            Dense(2, activation='elu')
        ))
        self.model.compile(
            optimizer=keras.optimizers.Adam,
            loss='binary_crossentropy',
            metrics='binary_accuracy'
        )

    def get_model(self):
        return self.model

    def train(self, x_train, y_train, x_val, y_val):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            mode='auto'
        )
        history = self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            shuffle=True,
            callbacks=[early_stopping],
            validation_data=(x_val, y_val)
        )
        return history

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
