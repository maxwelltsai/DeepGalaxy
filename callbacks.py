import tensorflow as tf 


class DataReshuffleCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, dg_inst):
        self.dg_inst = dg_inst
    
    def on_epoch_end(self, epoch, logs=None):
        # print('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch, logs['loss'], logs['sparse_categorical_accuracy']))
        if self.dg_inst.data_loading_mode > 0:
            if epoch % self.dg_inst.data_loading_mode == 0:
                self.dg_inst.load_data()
