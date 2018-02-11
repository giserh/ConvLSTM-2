#!/usr/bin/env python3

from read_radar import read_radar

import numpy as np

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

def main():
    rawdir = './fsl_20161018-22'
    data_set = read_radar(rawdir)
    data_set.generate_radarfsl()
    n_pixel = 30

    seq = Sequential()
    seq.add(ConvLSTM2D(filters=n_pixel, kernel_size=(3, 3),
                       input_shape=(None, n_pixel, n_pixel, 1),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=n_pixel, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=n_pixel, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=n_pixel, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))

    seq.compile(loss='mean_squared_error', optimizer='adadelta')

    # Train the network
    noisy_movies, shifted_movies = data_set.train_data, data_set.shifted_data
    seq.fit(noisy_movies[:1000], shifted_movies[:1000], batch_size=10,
            epochs=50, validation_split=0.05)

    #save model
    seq.save('ConvLSTM.h5')

if __name__ == '__main__':
    main()

