#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

from generate_lstm_data import generate_movies

batch_size = 10
timesteps = 15
shape = [40, 40]
kernel = [3, 3]
channels = 1
filters = 12

# Create a placeholder for videos.
inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])
print(inputs)
noisy_movies, shifted_movies = generate_movies()
print(np.shape(noisy_movies))

# Add the ConvLSTM step.
from ConvLSTMCell import ConvLSTMCell
cell = ConvLSTMCell(shape, filters, kernel)
outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)

# There's also a ConvGRUCell that is more memory efficient.
from ConvGRUCell import ConvGRUCell
cell = ConvGRUCell(shape, filters, kernel)
outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)

# It's also possible to enter 2D input or 4D input instead of 3D.
shape = [100]
kernel = [3]
inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])
cell = ConvLSTMCell(shape, filters, kernel)
outputs, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype=inputs.dtype)

shape = [50, 50, 50]
kernel = [1, 3, 5]
inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])
cell = ConvGRUCell(shape, filters, kernel)
outputs, state= tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype=inputs.dtype)
