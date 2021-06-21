import tensorflow as tf 
from datetime import datetime


class DataReshuffleCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, dg_inst):
        self.dg_inst = dg_inst # DeepGalaxy instance, to access the data loading routine
    
    def on_epoch_end(self, epoch, logs=None):
        # print('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch, logs['loss'], logs['sparse_categorical_accuracy']))
        if self.dg_inst.data_loading_mode > 0:
            if epoch % self.dg_inst.data_loading_mode == 0:
                self.dg_inst.load_data()


class TimingCallback(tf.keras.callbacks.Callback):

    def __init__(self, rank, n_workers, dg_inst):
        self.rank=rank
        self.n_workers = n_workers
        self.dg_inst = dg_inst # DeepGalaxy instance, to access the size of the training data (for calculating the Tput)
        self.logger = dg_inst.logger 

    def on_epoch_begin(self, epoch, logs=None):
        self.t_start = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        self.t_end = datetime.now()
        dt = (self.t_end - self.t_start).total_seconds()
        n_samples = self.dg_inst.y_train.shape[0]

        if self.rank == 0:
            msg = '[Performance] Epoch %d takes %0.2f seconds. Throughput: %0.2f images/sec (per node), %0.2f images/sec (total)' % (epoch, dt, n_samples/dt, n_samples/dt*self.n_workers) 
            
            if self.logger is None:
                print(msg)
            else:
                self.logger.info(msg)