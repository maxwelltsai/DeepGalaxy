"""
Deep Galaxy Training code.

Maxwell Cai (SURF), October - November 2019.

# Dynamic loading of training data
# Integrate the parallel version with the single-core version
"""



# import keras as K
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
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
import logging

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
        self.use_noise = True
        self.distributed_training = False
        self.multi_gpu_training = False
        self._multi_gpu_model = None
        self._n_gpus = 1
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
        # init_op = tf.initialize_all_variables()
        # init_op = tf.global_variables_initializer()
        # sess = tf.Session()
        # sess.run(init_op)

        # Check if GPUs are available
        # if tf.test.is_gpu_available():  # commented out since this test will cause a new session be created
        # allow growth
        # config = tf.compat.v1.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 1
        # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        # # config.log_device_placement = True  # to log device placement (on which device the operation ran)
        # sess = tf.compat.v1.Session(config=config)
        # tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras

        # Create logger
        self.logger = logging.getLogger('DeepGalaxyTrain')
        self.logger.setLevel(self.log_level)
        self.logger.addHandler(logging.FileHandler('train_log.txt'))
        if self.distributed_training is True:
            try:
                import horovod.tensorflow.keras as hvd
                # initialize horovod
                hvd.init()
                self.callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
                self.callbacks.append(hvd.callbacks.MetricAverageCallback())
                # self.callbacks = [hvd.BroadcastGlobalVariablesHook(0)]
                if hvd.rank() == 0:
                    self.logger.info('Parallel training enabled.')
                    self.logger.info('batch_size = %d, global_batch_size = %d, num_workers = %d\n' % (self.batch_size, self.batch_size*hvd.size(), hvd.size()))

                # Map an MPI process to a GPU (Important!)
                print('hvd_rank = %d, hvd_local_rank = %d' % (hvd.rank(), hvd.local_rank()))
                self.logger.info('hvd_rank = %d, hvd_local_rank = %d' % (hvd.rank(), hvd.local_rank()))

                # Bind a CUDA device to one MPI process (has no effect if GPUs are not used)
                os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())

                # # Horovod: pin GPU to be used to process local rank (one GPU per process)
                gpus = tf.config.experimental.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                # if gpus:
                    # tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
            except ImportError as identifier:
                print('Error importing horovod. Disabling distributed training.')
                self.distributed_training = False
        else:
            self.logger.info('Parallel training disabled.')
            self.logger.info('Batch_size = %d' % (self.batch_size))


    def load_data(self, data_fn, dset_name_pattern, camera_pos, test_size=0.2, random=True):
        if not self.distributed_training:
            self.logger.info('Loading the full dataset since distributed training is disabled ...')
            X, Y = self.data_io.load_all(data_fn, dset_name_pattern=dset_name_pattern, camera_pos=camera_pos)
        else:
            self.logger.info('Loading part of the dataset since distributed training is enabled ...')
            X, Y = self.data_io.load_partial(data_fn, dset_name_pattern=dset_name_pattern, camera_pos=camera_pos, hvd_size=hvd.size(), hvd_rank=hvd.rank())
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
        self.num_classes = np.unique(Y).shape[0]
        self.logger.debug('Number of classes: %d' % self.num_classes)

    def load_model(self):
        if not os.path.isfile('efn_b4.h5'):
            base_model = efn.EfficientNetB4(weights=None, include_top=True, input_shape=(self.input_shape[0], self.input_shape[1], 3), classes=self.num_classes)
            base_model.save('efn_b4.h5')
        else:
            base_model = tf.keras.models.load_model('efn_b4.h5', compile=False)
        print(base_model.summary())
        if not self.use_noise:
            # x = base_model.output
            # x = tf.keras.layers.GlobalAveragePooling2D()(x)
            # x = tf.keras.layers.Dropout(0.3)(x)
            # predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
            # model = tf.keras.models.Model(inputs = base_model.input, outputs = predictions)
            # model = tf.keras.models.Model(inputs = base_model.input, outputs = base_model.outputs)
            model = tf.keras.models.Sequential()
            # model.add(tf.keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1), input_shape=self.input_shape))  # commented out since tf.repeat does not exist before 1.15
            model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, 3, axis=-1), input_shape=self.input_shape))
            model.add(base_model)
            # model.add(tf.keras.layers.GlobalAveragePooling2D())
            # model.add(tf.keras.layers.Dropout(0.3))
            # model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        else:
            model = tf.keras.models.Sequential()
            # model.add(tf.keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1), input_shape=self.input_shape))  # commented out since tf.repeat does not exist before 1.15
            model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, 3, axis=-1), input_shape=self.input_shape))
            model.add(tf.keras.layers.GaussianNoise(0.5, input_shape=self.input_shape))
            model.add(base_model)
            # model.add(tf.keras.layers.GlobalAveragePooling2D(name="gap"))
            # model.add(tf.keras.layers.Dropout(0.3))
            # model.add(tf.keras.layers.Dense(self.num_classes, activation="softmax", name="fc_out"))

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
                          metrics=['sparse_categorical_accuracy'], experimental_run_tf_function=False)
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
                if self.use_noise is True:
                    self.model.save('model_hvd_bw_%d_B4_with_noise_n_p_%d.h5' % (self.input_shape[0], hvd.size()))
                else:
                    self.model.save('model_hvd_bw_%d_B4_no_noise_%d_nodes.h5' % (self.input_shape[0], hvd.size()))
        else:
            if self.use_noise is True:
                self.model.save('model_bw_%d_B4_with_noise.h5' % (self.input_shape[0]))
            else:
                self.model.save('model_bw_%d_B4_no_noise.h5' % (self.input_shape[0]))

    def finalize(self):
        pass


if __name__ == "__main__":
    dgtrain = DeepGalaxyTraining()
    dgtrain.distributed_training = True
    dgtrain.multi_gpu_training = False
    dgtrain.use_noise = True
    dgtrain.initialize()
    dgtrain.load_data('../output_bw_512.hdf5', dset_name_pattern='s_1_m_1*', camera_pos=[1,2,3])
    # dgtrain.load_data('../output_bw_512.hdf5', dset_name_pattern='s_*', camera_pos='*')
    # dgtrain.load_data('../output_bw_1024.hdf5', dset_name_pattern='s_*', camera_pos='*')
    dgtrain.load_model()
    dgtrain.fit()
    dgtrain.save_model()
    dgtrain.finalize()
