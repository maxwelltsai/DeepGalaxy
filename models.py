import random, re, math
import numpy as np, pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn

# from aug_layers import *
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation



def transform(image):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    # image = image[0]
    DIM = image.shape[1]
    print("image shapr", tf.shape(image), DIM)
    batch_size = tf.shape(image)
    XDIM = DIM%2 #fix for size 331

    rot = 15. * tf.random.normal(batch_size,dtype='float32')
    shr = 5. * tf.random.normal(batch_size,dtype='float32')
    h_zoom = 1.0 + tf.random.normal(batch_size,dtype='float32')/10.
    w_zoom = 1.0 + tf.random.normal(batch_size,dtype='float32')/10.
    h_shift = 16. * tf.random.normal(batch_size,dtype='float32')
    w_shift = 16. * tf.random.normal(batch_size,dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))
    idx2 = K.cast(idx2,dtype='int32')
    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image,tf.transpose(idx3))

    return tf.reshape(d,[4, DIM,DIM,3])


def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )

    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )

    # ZOOM MATRIX
    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )

    # SHIFT MATRIX
    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )

    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))

# ---------------

def get_base_model(base_model_name, input_shape, classes=1000, include_top=True, weights=None, pooling=None, **kwargs):
    if 'EfficientNet' in base_model_name:
        base_model = getattr(efn, base_model_name)(weights=weights, include_top=include_top, input_shape=input_shape, classes=classes, pooling=pooling, **kwargs)
    else:
        base_model = getattr(tf.keras.applications, base_model_name)(weights=weights, include_top=include_top, input_shape=input_shape, classes=classes, pooling=pooling, **kwargs)
    return base_model

def simple_keras_application(base_model_name, input_shape, classes, noise_stddev=0.0):
    base_model = get_base_model(base_model_name, (input_shape[0], input_shape[1], 3), classes)
    model = tf.keras.models.Sequential()

    if input_shape[-1] == 1:
        # single channel images. Repeat channel needed
        model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, 3, axis=-1), input_shape=input_shape, name='repeat_channels'))
    if noise_stddev > 0:
        # generate random Gaussian noise according to the stddev
        model.add(tf.keras.layers.GaussianNoise(noise_stddev, input_shape=input_shape))
    model.add(base_model)
    return model

def simple_keras_application_with_imagenet_weights(base_model_name, input_shape, classes, noise_stddev=0.0):
    base_model_with_top = get_base_model(base_model_name, (input_shape[0], input_shape[1], 3), classes, include_top=True, weights=None)
    base_model_no_top = get_base_model(base_model_name, (input_shape[0], input_shape[1], 3), classes, include_top=False, weights='imagenet')

    model = tf.keras.models.Sequential()
    if input_shape[-1] == 1:
        # single channel images. Repeat channel needed
        model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, 3, axis=-1), input_shape=input_shape, name='repeat_channels'))

    if noise_stddev > 0:
        # generate random Gaussian noise according to the stddev
        # TODO: replace with augmentation method from Kaggle
        model.add(tf.keras.layers.GaussianNoise(noise_stddev, input_shape=input_shape))

    model.add(base_model_no_top)
    for i in range(len(base_model_no_top.layers), len(base_model_with_top.layers)):
        model.add(base_model_with_top.layers[i])
    return model

