import tensorflow as tf 
from datetime import datetime


class DataReshuffleCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, dg_inst):
        self.dg_inst = dg_inst
    
    def on_epoch_end(self, epoch, logs=None):
        # print('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch, logs['loss'], logs['sparse_categorical_accuracy']))
        if self.dg_inst.data_loading_mode > 0:
            if epoch % self.dg_inst.data_loading_mode == 0:
                self.dg_inst.load_data()


class TimingCallback(tf.keras.callbacks.Callback):

    def __init__(self, rank):
        self.rank=rank

    def on_epoch_begin(self, epoch, logs=None):
        self.t_start = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        self.t_end = datetime.now()
        dt = (self.t_end - self.t_start).total_seconds()

        if self.rank == 0:
            print('\t[Timing] Epoch %d takes %0.2f seconds' % (epoch, dt))