"""
Deep Galaxy Training code.

Maxwell Cai (SURF), October - November 2019.

# Dynamic loading of training data
# Integrate the parallel version with the single-core version
"""



# import keras as K
import tensorflow as tf
# import efficientnet.keras_custom as efn
import efficientnet.tfkeras as efn
from skimage.io import imread
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
# from efficientnet.keras_custom import center_crop_and_resize, preprocess_input
# from keras.applications.imagenet_utils import decode_predictions
# from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from data_io import DataIO
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import psutil
import socket
from keras.callbacks import Callback
import time
import argparse 

try:
    import horovod.keras as hvd
except ImportError as ie:
    pass

tf.compat.v1.disable_eager_execution()

class DeepGalaxyTraining(object):

    def __init__(self):
        self.data_io = DataIO()
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.num_classes = 0
        self.epochs = 12
        self.batch_size = 4
        self.use_noise = False 
        self.distributed_training = False 
        self.multi_gpu_training = False
        self._multi_gpu_model = None
        self._n_gpus = 1
        self.callbacks = []
        self.f_usage = None
        self.input_shape = (512, 512, 3)  # (256, 256, 3)
        self._t_start = 0
        self._t_end = 0

    def get_flops(self, model):
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()

        # We use the Keras session graph in the call to the profiler.
        flops = tf.profiler.profile(graph=tf.keras.backend.get_session().graph,
                                    run_meta=run_meta, cmd='op', options=opts)

        return flops.total_float_ops  # Prints the "flops" of the model.

    def initialize(self):
        # init_op = tf.initialize_all_variables()
        # init_op = tf.global_variables_initializer()
        # sess = tf.Session()
        # sess.run(init_op)

        if self.distributed_training is True:
            try:
                import horovod.keras as hvd
                # initialize horovod
                hvd.init()
                self.callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
                # self.callbacks = [hvd.BroadcastGlobalVariablesHook(0)]
                if hvd.rank() == 0:
                    self.f_usage = open('usage_%d_2048.txt' % hvd.size(), 'w')
                    # self.f_usage.write('batch_size: %d, global_batch_size: %d, num_workers, %d, N_train_batches: %d, N_test_batches: %d\n' % (self.batch_size, global_batch_size, hvd.size(), train_batches, 0))
                    self.f_usage.flush()
            except ImportError as identifier:
                print('Error importing horovod. Disabling distributed training.')
                self.distributed_training = False
        else:
            self.f_usage = open('usage_2048.txt', 'w')
            # self.f_usage.write('batch_size: %d, N_train_batches: %d, N_test_batches: %d\n' % (self.batch_size, train_batches, 0))
            self.f_usage.flush()


    def load_data(self, data_fn, dset_name_pattern, camera_pos, test_size=0.2, random=True):
        if not self.distributed_training:
            print('load all')
            X, Y = self.data_io.load_all(data_fn, dset_name_pattern=dset_name_pattern, camera_pos=camera_pos)
        else:
            print('load partial')
            X, Y = self.data_io.load_partial(data_fn, dset_name_pattern=dset_name_pattern, camera_pos=camera_pos, hvd_size=hvd.size(), hvd_rank=hvd.rank())
        print(X.shape, Y.shape)
        if test_size > 0:
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test
        else:
            self.x_train = X
            self.y_train = Y
        self.num_classes = np.unique(Y).shape[0]
        print('This is Y', X)
        print('This is Y', Y)

    def load_model(self):
        if not os.path.isfile('efn_b4.h5'):
            base_model = efn.EfficientNetB4(weights=None, include_top=False, input_shape=self.input_shape, classes=self.num_classes)
            base_model.save('efn_b4.h5')
        else:
            base_model = tf.keras.models.load_model('efn_b4.h5')
        if not self.use_noise:
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
            model = tf.keras.models.Model(inputs = base_model.input, outputs = predictions)
        else:
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.GaussianNoise(0.5, input_shape=self.input_shape))
            model.add(base_model)
            model.add(tf.keras.layers.GlobalAveragePooling2D(name="gap"))
            model.add(tf.keras.layers.Dropout(0.3))
            model.add(tf.keras.layers.Dense(self.num_classes, activation="softmax", name="fc_out"))

        if self.distributed_training is True:
            # opt = K.optimizers.SGD(0.001 * hvd.size())
            # opt = tf.keras.optimizers.Adam(hvd.size())
            opt = tf.keras.optimizers.Adadelta(1.0 * hvd.size())
            # Horovod: add Horovod Distributed Optimizer.
            opt = hvd.DistributedOptimizer(opt)
        else:
            opt = tf.keras.optimizers.Adam()

        if self.multi_gpu_training is True:
            # probe the number of GPUs
            from tensorflow.python.client import device_lib
            local_device_protos = device_lib.list_local_devices()
            gpu_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
            self._n_gpus = len(gpu_list)
            print('Parallalizing the model on %d GPUs...' % self._n_gpus)
            parallel_model = tf.keras.utils.multi_gpu_model(model, gpus=self._n_gpus)
            parallel_model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                                   optimizer=opt,
                                   metrics=['sparse_categorical_accuracy'])
            self._multi_gpu_model = parallel_model
            self.model = model
            print(parallel_model.summary())
        else:
            model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                          optimizer=opt,
                          metrics=['sparse_categorical_accuracy'])
            self.model = model
            # print(model.summary())

    def fit(self):
        if self.distributed_training is True:
            try:
                # print('len(train_iter)', len(train_iter))
                # if hvd.rank() == 0:
                    # self.f_usage.write('len(train_iter) = %d, x_train.shape=%s\n' % (len(train_iter), x_train.shape))
                self._t_start = datetime.now()
                self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size,
                               epochs=self.epochs,
                               callbacks=self.callbacks,
                               verbose=1 if hvd.rank()==0 else 0,
                               validation_data=(self.x_test, self.y_test))
                self._t_end = datetime.now()
                # train_gen = ImageDataGenerator()
                # train_iter = train_gen.flow(self.x_train, self.y_train, batch_size=self.batch_size)
                # test_gen = ImageDataGenerator()
                # test_iter = test_gen.flow(self.x_test, self.y_test, batch_size=self.batch_size)
                # self.model.fit_generator(train_iter,
                #     # batch_size=batch_size,
                #     steps_per_epoch=len(train_iter) // hvd.size(),
                #     epochs=self.epochs,
                #     callbacks=self.callbacks,
                #     verbose=1 if hvd.rank() == 0 else 0,
                #     validation_data=test_gen.flow(self.x_test, self.y_test, self.batch_size),
                #     validation_steps=len(test_iter) // hvd.size())
                
            except KeyboardInterrupt:
                print('Terminating due to Ctrl+C...')
            finally:
                print("On hostname {0} - After training using {1} GB of memory".format(socket.gethostname(), psutil.Process(os.getpid()).memory_info()[0]/1024/1024/1024))
                self._t_end = datetime.now()
                if hvd.rank() == 0:
                    self.f_usage.write("On hostname {0} - After training using {1} GB of memory\n".format(socket.gethostname(), psutil.Process(os.getpid()).memory_info()[0]/1024/1024/1024))
                    self.f_usage.write('Time is now %s\n' % datetime.now())
                    # self.f_usage.write('Elapsed time %s\n' % (t_end-t_start))
                    self.f_usage.flush()
                # print('Elapsed time:', t_end-t_start)
        else:
            try:
                if self.multi_gpu_training is True:
                    self._t_start = datetime.now()
                    self._multi_gpu_model.fit(self.x_train, self.y_train, 
                                              batch_size=self.batch_size * self._n_gpus, 
                                              epochs=self.epochs,
                                            #   callbacks=self.callbacks, 
                                              verbose=1, 
                                              validation_data=(self.x_test, self.y_test))
                    self._t_end = datetime.now()
                else:
                    self._t_start = datetime.now()
                    self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, 
                                   epochs=self.epochs,
                                #    callbacks=self.callbacks, 
                                   verbose=1, 
                                   validation_data=(self.x_test, self.y_test))
                    self._t_end = datetime.now()
            except KeyboardInterrupt:
                pass
            finally:
                self._t_end = datetime.now()
                print('Elapsed time:', self._t_end - self._t_start)
                print('Saving model...')
        print(self.get_flops(self.model))

    def save_model(self):
        if self.distributed_training is True:
            if hvd.rank() == 0:
                if self.use_noise is True:
                    self.model.save('model_hvd_bw_%d_B4_with_noise_%d_nodes.h5' % (image_dim, hvd.size()/2))
                else:
                    self.model.save('model_hvd_bw_%d_B4_no_noise_%d_nodes.h5' % (image_dim, hvd.size()/2))
        else:
            if self.use_noise is True:
                self.model.save('model_bw_%d_B4_with_noise.h5' % (image_dim))
            else:
                self.model.save('model_bw_%d_B4_no_noise.h5' % (image_dim))

    def finalize(self):
        self.f_usage.close()


if __name__ == "__main__":
    dgtrain = DeepGalaxyTraining()
    dgtrain.distributed_training = True  
    dgtrain.multi_gpu_training = False
    dgtrain.initialize()
    dgtrain.load_data('../output_bw_512.hdf5', dset_name_pattern='s_1_m_1*', camera_pos=[1,2,3])
    dgtrain.load_model()
    dgtrain.fit()
    dgtrain.save_model()
    dgtrain.finalize()
