import tensorflow as tf 
import efficientnet.tfkeras as efn


def get_base_model(base_model_name, input_shape, classes=1000, include_top=True, weights=None, pooling=None, **kwargs):
    if 'EfficientNet' in base_model_name:
        base_model = getattr(efn, base_model_name)(weights=weights, include_top=include_top, input_shape=input_shape, classes=classes, pooling=pooling, **kwargs)
    else:
        base_model = getattr(tf.keras.applications, base_model_name)(weights=weights, include_top=include_top, input_shape=input_shape, classes=classes, pooling=pooling, **kwargs)
    return base_model

def simple_keras_application(base_model_name, input_shape, classes, noise_stddev=0.0):
    base_model = get_base_model(base_model_name, (input_shape[0], input_shape[1], 3), classes)
    model = tf.keras.models.Sequential()
    
    # Data augmentation
    # model.add(tf.keras.layers.experimental.preprocessing.RandomFlip())
    # model.add(tf.keras.layers.experimental.preprocessing.RandomRotation(0.1))
    # model.add(tf.compat.v1.keras.preprocessing.image.)
    # model.add(tf.compat.v1.keras.preprocessing.image.random_rotation(rg=0.1))
    # model.add(tf.keras.layers.Lambda(lambda x: tf.compat.v1.keras.preprocessing.image.random_rotation(x, rg=10)))
    
    if input_shape[-1] == 1:
        # single channel images. Repeat channel needed
        model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, 3, axis=-1), input_shape=input_shape))
    if noise_stddev > 0: 
        # generate random Gaussian noise according to the stddev
        model.add(tf.keras.layers.GaussianNoise(noise_stddev, input_shape=input_shape))
    model.add(base_model)
    return model 

def simple_keras_application_with_imagenet_weight(base_model_name, input_shape, classes, noise_stddev=0.0):
    base_model_with_top = get_base_model(base_model_name, (input_shape[0], input_shape[1], 3), classes, include_top=True, weights=None)
    base_model_no_top = get_base_model(base_model_name, (input_shape[0], input_shape[1], 3), classes, include_top=False, weights='imagenet')
    
    model = tf.keras.models.Sequential()
    if input_shape[-1] == 1:
        # single channel images. Repeat channel needed
        model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, 3, axis=-1), input_shape=input_shape))
    if noise_stddev > 0: 
        # generate random Gaussian noise according to the stddev
        model.add(tf.keras.layers.GaussianNoise(noise_stddev, input_shape=input_shape))
    
    # Freeze the ImageNet weights
    # for layer in base_model_no_top.layers:
    #     layer.trainable = False
    
    model.add(base_model_no_top)
    for i in range(len(base_model_no_top.layers), len(base_model_with_top.layers)):
        model.add(base_model_with_top.layers[i])
    return model 
    