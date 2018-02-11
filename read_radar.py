#!/usr/bin/env python3

import time
import os

import numpy as np

class read_radar(object):

    def __init__(self, data_dir):
        self.rawdir = data_dir
        self.saved_file_name = 'rawdata.npz'
        self.ntimes = 0
        self.row = None
        self.col = None
        self.shift_len = 1
        self.train_data = None
        self.shifted_data = None

    def read_file(self, file_name):
        print('Run task %s (%s)...' % (file_name, os.getpid()))
        start = time.time()
        file_data = []
        with open(file_name) as f:
            line = f.readline()
            while line:
                line = line.strip('\n')
                file_data.append(line.split(','))
                line = f.readline()
        end = time.time()
        print('Task %s runs %0.2f seconds.' % (file_name, (end - start)))
        return np.array(file_data, dtype=float)[40:70, 40:70]

    def generate_radarfsl(self):
        if os.path.isfile(self.saved_file_name):
            self.loadArray()
        else:
            datas = []
            with open('./file_list', 'r') as file_list:
                for line in file_list.readlines():
                    line = line.strip('\n')
                    file_name = os.path.join(self.rawdir, line)
                    d = self.read_file(file_name)
                    datas.append(d)
                    self.ntimes += 1

            datas_array = np.array(datas)

            self.row, self.col = np.shape(datas[0])[0], np.shape(datas[0])[1]

            self.train_data = np.zeros((self.ntimes-24, 24,
                self.row, self.col, 1))

            self.shifted_data = np.zeros((self.ntimes-24, 24,
                self.row, self.col, 1))

            print(np.shape(self.train_data))
            print(np.shape(self.shifted_data))

            for i in range(24):
                print(i)
                self.train_data[:,i,:,:,0] = datas_array[i:i+self.ntimes-24]
                self.shifted_data[:,i,:,:,0] = datas_array[i+1:i+1+self.ntimes-24]

#            self.train_data[self.train_data < 3] = 0 
#            self.shifted_data[self.shifted_data < 3] = 0 
            self.shifted_data = self.shifted_data / 9.0
            self.train_data = self.train_data / 9.0

            self.saveArray()

    def saveArray(self):
        np.savez(self.saved_file_name , X=self.train_data,
                                        y=self.shifted_data)

    def loadArray(self):
        print('reading datas from saved file')
        arch = np.load(self.saved_file_name)
        self.train_data = arch['X']
        self.shifted_data = arch['y']
        print('end reading data')
        print(np.shape(self.train_data))
        print(np.shape(self.shifted_data))


if __name__ == '__main__':
    rawdir = 'fsl_20161018-22'
    data_set = read_radar(rawdir)
    data_set.generate_radarfsl()

