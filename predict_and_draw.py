#!/usr/bin/env python3

import numpy as np
import pylab as plt

from keras.models import load_model
from read_radar import read_radar

def main():

    model = load_model('./ConvLSTM.h5')

    rawdir = './fsl_20161018-22'
    data_set = read_radar(rawdir)
    data_set.generate_radarfsl()
    noisy_movies, shifted_movies = data_set.train_data, data_set.shifted_data

    # Testing the network on one movie
    # feed it with the first 7 positions and then
    # predict the new positions
    which = 40
    track = noisy_movies[which][:7, ::, ::, ::]
    
    for j in range(25):
        new_pos = model.predict(track[np.newaxis, ::, ::, ::, ::])
        new = new_pos[::, -1, ::, ::, ::]
        track = np.concatenate((track, new), axis=0)
    
    # And then compare the predictions
    # to the ground truth
    track2 = noisy_movies[which][::, ::, ::, ::]
    for i in range(24):
        fig = plt.figure(figsize=(10, 5))
    
        ax = fig.add_subplot(121)
    
        if i >= 7:
            ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
        else:
            ax.text(1, 3, 'Inital trajectory', fontsize=20)
    
        toplot = track[i, ::, ::, 0]
    
        plt.imshow(toplot)
        ax = fig.add_subplot(122)
        plt.text(1, 3, 'Ground truth', fontsize=20)
    
        toplot = track2[i, ::, ::, 0]
        if i >= 2:
            toplot = shifted_movies[which][i - 1, ::, ::, 0]
    
        plt.imshow(toplot)
        plt.savefig('%i_animate.png' % (i + 1))

if __name__ == '__main__':
    main()
