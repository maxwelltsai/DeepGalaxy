"""
Deep Galaxy Training code.

Maxwell Cai (SURF), October 2019 - May 2020.

# Dynamic loading of training data
# Integrate the parallel version with the single-core version
"""



import tensorflow as tf
import efficientnet.tfkeras as efn
from skimage.io import imread
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
import numpy as np
import pandas as pd
from data_io_new import DataIO
from models import *
from callbacks import DataReshuffleCallback
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import psutil
import socket
from tensorflow.keras.callbacks import Callback
import time
import argparse
import logging
from tensorflow.keras.mixed_precision import experimental as mixed_precision

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
        self.epochs = 50
        self.batch_size = 4
        self.noise_stddev = 0.3
        self.learning_rate = 1.0  # depends on the optimizer
        self.data_loading_mode = 0 
        self.base_model_name = 'EfficientNetB4'
        self.distributed_training = False
        self.multi_gpu_training = False
        self._gpu_memory_allow_growth = False
        self._gpu_memory_fraction = None  # a number greater than 1 means that unified memory will be used; set to None for automatic handling
        self._multi_gpu_model = None
        self.debug_mode = False 
        self._n_gpus = 1
        self._data_fn = None
        self._dset_name_pattern = None 
        self._camera_pos = None 
        self.callbacks = []
        self.logger = None
        self.log_level = logging.DEBUG
        self.input_shape = (512, 512, 3)  # (256, 256, 3)
        self._t_start = 0
        self._t_end = 0

    def get_flops(self, model):
        # run_meta = tf.RunMetadata()  # commented out since it doesn't work in TF2
        run_meta = tf.compat.v1.RunMetadata()
        # opts = tf.profiler.ProfileOptionBuilder.float_operation()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

        # We use the Keras session graph in the call to the profiler.
        flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session().graph,
                                    run_meta=run_meta, cmd='op', options=opts)

        return flops.total_float_ops  # Prints the "flops" of the model.

    def initialize(self):
        # set TF image format
        tf.keras.backend.set_image_data_format('channels_last')
        
        # set data pipeline loading/partitioning strategy
        if self.data_loading_mode < 1:
            # -1: load full dataset on node; 0: load partial dataset, but don't shuffle
            self.data_io._partitioning_strategy = self.data_loading_mode 
        else:
            # When `data_loading_mode` > 0, a node loads a partial dataset randomly sampled from the 
            #       full dataset, and it will do the loading per `data_load_model` epochs. In this case,
            #       the `_partitioning_strategy` of `data_io` is set to 1 to ensure random sampling.
            self.data_io._partitioning_strategy = 1
        
        # set up distributed training facility 
        if self.distributed_training is True:
            try:
                import horovod.tensorflow.keras as hvd
                # initialize horovod
                hvd.init()
                if hvd.rank() == 0:
                    # Create logger
                    self.logger = logging.getLogger('DeepGalaxyTrain')
                    self.logger.setLevel(self.log_level)
                    self.logger.addHandler(logging.FileHandler('train_log.txt'))
                    self.logger.info('Parallel training enabled.')
                    self.logger.info('batch_size = %d, global_batch_size = %d, num_workers = %d\n' % (self.batch_size, self.batch_size*hvd.size(), hvd.size()))

                # Map an MPI process to a GPU (Important!)
                print('hvd_rank = %d, hvd_local_rank = %d' % (hvd.rank(), hvd.local_rank()))
                if hvd.rank() == 0:
                    self.logger.info('hvd_rank = %d, hvd_local_rank = %d' % (hvd.rank(), hvd.local_rank()))

                # Bind a CUDA device to one MPI process (has no effect if GPUs are not used)
                os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())
                
                # Add callbacks
                self.callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
                self.callbacks.append(hvd.callbacks.MetricAverageCallback())
                self.callbacks.append(DataReshuffleCallback(self))
                if self.debug_mode is True:
                    if hvd.rank() == 0:
                        if not os.path.isdir('logs'):
                            os.mkdir('logs')
                        if not os.path.isdir('checkpoints'):
                            os.mkdir('checkpoints')
                        self.callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
                                                                            update_freq='epoch', profile_batch=2, embeddings_freq=0, embeddings_metadata=None))
                        self.callbacks.append(tf.keras.callbacks.ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_weights_only=True))


                # Configure GPUs (if any)
                gpus = tf.config.experimental.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, self._gpu_memory_allow_growth)
                    tf.config.experimental.set_visible_devices(gpu, 'GPU')
                if self._gpu_memory_fraction is not None:
                    config = tf.compat.v1.ConfigProto()
                    config.gpu_options.per_process_gpu_memory_fraction = self._gpu_memory_fraction
                    session = tf.compat.v1.InteractiveSession(config=config)
                    
            except ImportError as identifier:
                print('Error importing horovod. Disabling distributed training.')
                self.distributed_training = False
                self.logger = logging.getLogger('DeepGalaxyTrain')
                self.logger.setLevel(self.log_level)
                self.logger.addHandler(logging.FileHandler('train_log.txt'))
                self.logger.info('Parallel training disabled.')
                self.logger.info('Batch_size = %d' % (self.batch_size))
        else:
            self.callbacks.append(DataReshuffleCallback(self))
            if self.debug_mode is True:
                if not os.path.isdir('logs'):
                    os.mkdir('logs')
                if not os.path.isdir('checkpoints'):
                    os.mkdir('checkpoints')
                self.callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
                                                                    update_freq='epoch', profile_batch=2, embeddings_freq=0, embeddings_metadata=None))
                self.callbacks.append(tf.keras.callbacks.ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_weights_only=True))
            # Create logger
            self.logger = logging.getLogger('DeepGalaxyTrain')
            self.logger.setLevel(self.log_level)
            self.logger.addHandler(logging.FileHandler('train_log.txt'))
            self.logger.info('Parallel training disabled.')
            self.logger.info('Batch_size = %d' % (self.batch_size))


    def load_data(self, data_fn=None, dset_name_pattern=None, camera_pos=None, test_size=0.2, random=True):
        if data_fn is not None and dset_name_pattern is not None and camera_pos is not None:
            self._data_fn = data_fn
            self._dset_name_pattern = dset_name_pattern
            self._camera_pos = camera_pos
        elif self._data_fn is not None and self._dset_name_pattern is not None and self._camera_pos is not None:
            data_fn = self._data_fn
            dset_name_pattern = self._dset_name_pattern
            camera_pos = self._camera_pos
        else:
            raise ValueError('The data_fn, dset_name_pattern, camera_pos arguments should be specified.')
            
        if not self.distributed_training:
            self.logger.info('Loading the full dataset since distributed training is disabled ...')
            X, Y, self.num_classes = self.data_io.load_all(data_fn, dset_name_pattern=dset_name_pattern, camera_pos=camera_pos)
            self.logger.debug('Shape of X: %s' % str(X.shape))
            self.logger.debug('Shape of Y: %s' % str(Y.shape))
        else:
            X, Y, self.num_classes = self.data_io.load_partial(data_fn, dset_name_pattern=dset_name_pattern, camera_pos=camera_pos, hvd_size=hvd.size(), hvd_rank=hvd.rank())
            if hvd.rank() == 0:
                self.logger.info('Loading part of the dataset since distributed training is enabled ...')
                self.logger.debug('Shape of X: %s' % str(X.shape))
                self.logger.debug('Shape of Y: %s' % str(Y.shape))

        # update the input_shape setting according to the loaded data
        self.input_shape = X.shape[1:]

        if test_size > 0:
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test
        else:
            self.x_train = X
            self.y_train = Y
        if not self.distributed_training:
            self.logger.debug('Number of classes: %d' % self.num_classes)
        else:
            if hvd.rank() == 0:
                self.logger.debug('Number of classes: %d' % self.num_classes)

    def load_model(self):

        # model = simple_keras_application(self.base_model_name, input_shape=self.input_shape, classes=self.num_classes)
        model = simple_keras_application_with_imagenet_weight(self.base_model_name, input_shape=self.input_shape, classes=self.num_classes)

        if self.distributed_training is True:
            # opt = K.optimizers.SGD(0.001 * hvd.size())
            # opt = tf.keras.optimizers.Adam(hvd.size())
            opt = tf.keras.optimizers.Adadelta(self.learning_rate * hvd.size())
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
            if self.distributed_training is True:
                if hvd.rank() == 0:
                    print(model.summary())
            else:
                print(model.summary())

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
                    self.logger.info("On hostname {0} - After training using {1} GB of memory\n".format(socket.gethostname(), psutil.Process(os.getpid()).memory_info()[0]/1024/1024/1024))
                    self.logger.info('Time is now %s\n' % datetime.now())
                    # self.f_usage.write('Elapsed time %s\n' % (t_end-t_start))
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
                if self.noise_stddev > 0 is True:
                    self.model.save('model_%d_%s_noise_np_%d.h5' % (self.input_shape[0], self.base_model_name, hvd.size()))
                else:
                    self.model.save('model_%d_%s_np_%d.h5' % (self.input_shape[0], self.base_model_name, hvd.size()))
        else:
            if self.noise_stddev > 0 is True:
                self.model.save('model_%d_%s_noise.h5' % (self.input_shape[0], self.base_model_name))
            else:
                self.model.save('model_%d_%s.h5' % (self.input_shape[0], self.base_model_name))

    def finalize(self):
        pass


