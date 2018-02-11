import numpy as np

class generate_sin_datas(object):

    def __init__(self, batch_size=128, n_input=1000):
        self.batch_size = batch_size
        self.n_input = n_input

    def next_batch(self):
        datas = np.ndarray(shape=(self.batch_size, self.n_input))
        datas[:,0] = 10 * np.random.rand(self.batch_size) 
        for i, val in enumerate(datas[:,0]):
            datas[i,:] = np.sin(np.linspace(val, val+100, self.n_input, dtype=np.float32)) 
        x = datas[:,0:-1]
        y = datas[:,-1]
        return x, y

def generate_movies(batch_size=10,shape=[80,80], n_frames=15):
    row, col = shape
    noisy_movies = np.zeros((batch_size, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((batch_size, n_frames, row, col, 1),
                              dtype=np.float)

    for i in range(batch_size):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)

        for j in range(n):
            # Initial position
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # Direction of motion
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # Size of the square
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[i, t, x_shift - w: x_shift + w,
                             y_shift - w: y_shift + w, 0] += 1

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the network to be robust and still
                # consider it as a pixel belonging to a square.
                if np.random.randint(0, 2):
                    noise_f = (-1)**np.random.randint(0, 2)
                    noisy_movies[i, t,
                                 x_shift - w - 1: x_shift + w + 1,
                                 y_shift - w - 1: y_shift + w + 1,
                                 0] += noise_f * 0.1

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w,
                               y_shift - w: y_shift + w, 0] += 1

    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies
