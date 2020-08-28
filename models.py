import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras import layers
from common_blocks import * 
# from image_preprocessing import *
# from tensorflow.keras.layers.experimental.preprocessing import *
# from aug_layers import *

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

def simple_keras_application_with_imagenet_weights(base_model_name, input_shape, classes, noise_stddev=0.0):
    base_model_with_top = get_base_model(base_model_name, (input_shape[0], input_shape[1], 3), classes, include_top=True, weights=None)
    base_model_no_top = get_base_model(base_model_name, (input_shape[0], input_shape[1], 3), classes, include_top=False, weights='imagenet')

    model = tf.keras.models.Sequential()
    if input_shape[-1] == 1:
        # single channel images. Repeat channel needed
        model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, 3, axis=-1), input_shape=input_shape))
    if noise_stddev > 0:
        # generate random Gaussian noise according to the stddev
        model.add(tf.keras.layers.GaussianNoise(noise_stddev, input_shape=input_shape))
    # model.add(RandomRotation(0.1))
    # Freeze the ImageNet weights
    # for layer in base_model_no_top.layers:
    #     layer.trainable = False

    model.add(base_model_no_top)
    for i in range(len(base_model_no_top.layers), len(base_model_with_top.layers)):
        model.add(base_model_with_top.layers[i])
    return model

# ---------------------------------------------------------------------
#  Unet Decoder
# ---------------------------------------------------------------------

def build_fcae(
        backbone,
        decoder_block,
        skip_connection_layers,
        input_shape,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='tanh',
        use_batchnorm=True,
):
    print('the input shape is', input_shape)
    if input_shape[-1] == 1 and backbone.input_shape[-1] > 1:
        input_ = tf.keras.layers.Input(shape=input_shape, name='input')
        x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, 3, axis=-1), name='repeat_channel')(input_)
    else:
        input_ = backbone.input
        x = input_ 
        
    x = backbone(x)
    encoded_features = layers.Activation('linear', name='backbone_activation')(x)
    
    decoder_inputs = layers.Input(shape=encoded_features.shape[1:])
    x = decoder_inputs
    
    # extract skip connections
    # skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
            #   else backbone.get_layer(index=i).output for i in skip_connection_layers])
    skips = []

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    encoder = models.Model(input_, encoded_features, name='conv_encoder')
    decoder = models.Model(decoder_inputs, x, name='conv_decoder')
    model = models.Sequential([
        encoder,
        decoder
    ])
    return model, encoder, decoder


# ---------------------------------------------------------------------
#  FCAE Model
# ---------------------------------------------------------------------

def FCAE(
        backbone=None,
        backbone_name='VGG16',
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights=None,
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        **kwargs
):
    """ FCAE is a fully convolution autoencoder neural network.
    Args:
        backbone: name or tf.keras.model.Model instance of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
            case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
            able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
            (e.g. ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
            Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
            layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
        decoder_block_type: one of blocks with following layers structure:
            - `upsampling`:  ``UpSampling2D`` -> ``Conv2D`` -> ``Conv2D``
            - `transpose`:   ``Transpose2D`` -> ``Conv2D``
        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
    Returns:
        ``keras.models.Model``: **Unet**
    """

    # global backend, layers, models, keras_utils
    # submodule_args = filter_keras_submodules(kwargs)
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)

    if decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))

    if backbone is None and backbone_name is not None:
        # if the backbone is a name, construct it 
        if 'EfficientNet' in backbone_name:
            backbone = getattr(efn, backbone_name)(weights=encoder_weights, include_top=False, input_shape=input_shape, classes=classes, pooling=None, **kwargs)
        else:
            backbone = getattr(tf.keras.applications, backbone_name)(weights=encoder_weights, include_top=False, input_shape=input_shape, classes=classes, pooling=None, **kwargs)
    else:
        # it is a Model instance. Just directly use it.
        pass 
    # backbone = Backbones.get_backbone(
    #     backbone_name,
    #     input_shape=input_shape,
    #     weights=encoder_weights,
    #     include_top=False,
    #     **kwargs,
    # )
    print(backbone)

    if encoder_features == 'default':
        encoder_features = get_feature_layers(backbone_name, n=4)

    model, encoder, decoder = build_fcae(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        input_shape=input_shape,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model, encoder, decoder


def VAE(input_shape, latent_dim=16, intermediate_dim=256, weights=None):
    inputs = layers.Input(shape=input_shape)
    # norm_factor = tf.math.reduce_max(inputs).numpy()
    # norm_factor = 1.0
    # inputs_normed = inputs / norm_factor
    # flattened = layers.GlobalAveragePooling2D(name='gap')(inputs) # don't use GAP, otherwise the tsne(z) is chaotic
    # flattened = layers.Flatten()(inputs_normed)
    # inputs_normed = layers.LayerNormalization(axis=[1,2,3])(inputs)
    flattened = layers.Flatten()(inputs)
    h = layers.Dense(intermediate_dim, name='intermediate_dense_encoder_1', activation='tanh')(flattened)
    h = layers.Dense(intermediate_dim, name='intermediate_dense_encoder_2', activation='linear')(h)
    z_mu = layers.Dense(latent_dim, name='z_mu')(h)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(h)

    # z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
    # z_sigma = layers.Lambda(lambda t: backend.exp(.5*t))(z_log_var)
    # eps = backend.random_normal(stddev=1.0, shape=tf.shape(z_sigma))
    # z_eps = layers.Multiply()([z_sigma, eps])
    # z = layers.Add(name='z')([z_mu, z_eps])
    z = KLDivergenceLayer(name='z')([z_mu, z_log_var])
    
    # encoder = models.Model(inputs, [z_mu, z_sigma, z], name='encoder')
    
    h = layers.Dense(intermediate_dim, input_shape=input_shape, name='intermediate_dense_decoder_1', activation='linear')(z)
    h = layers.Dense(intermediate_dim, input_shape=input_shape, name='intermediate_dense_decoder_2', activation='linear')(h)
    
    h = layers.Dense(tf.reduce_prod(input_shape), activation='linear', name='expand')(h)
    # h = layers.Dense(flattened.shape[1], activation='linear')(h)
    # h = layers.Reshape((1,1,flattened.shape[1]))(h)
    # h = layers.UpSampling2D((2,2))(h)
    # h = layers.Conv2DTranspose(filters=flattened.shape[1], kernel_size=(3,3), padding='same', activation='relu')(h)
    # h = layers.UpSampling2D((2,2))(h)
    # h = layers.Conv2DTranspose(filters=flattened.shape[1], kernel_size=(3,3), padding='same', activation='relu')(h)
    # h = layers.UpSampling2D((2,2))(h)
    # h = layers.Conv2DTranspose(filters=flattened.shape[1], kernel_size=(3,3), padding='same', activation='relu')(h)
    # h = layers.UpSampling2D((2,2))(h)
    # reconstructed_normed = layers.Conv2DTranspose(filters=flattened.shape[1], kernel_size=(3,3), padding='same', activation='sigmoid')(h)
    reconstructed = layers.Reshape(input_shape, name='restore_shape')(h)
    # reconstructed = layers.Lambda(lambda x: x * norm_factor, name='reconstructed')(reconstructed_normed)
    
    # decoder = models.Model(intermediate_dense(z_inputs), h)
    vae = models.Model(inputs, reconstructed)
    print(vae.summary())
    
    # load pretrained weights if specified
    if weights is not None:
        vae.load_weights(weights)
    encoder = models.Model(vae.inputs, [vae.get_layer('z_mu').output, vae.get_layer('z_log_var').output, vae.get_layer('z').output])
    
    decoder = tf.keras.models.Sequential()
    decoder.add(tf.keras.layers.Input(shape=latent_dim,))
    decoder.add(vae.get_layer('intermediate_dense_decoder_1'))
    # decoder.add(vae.get_layer('intermediate_dense_decoder_2'))
    decoder.add(vae.get_layer('expand'))
    decoder.add(vae.get_layer('restore_shape'))
    # decoder.add(vae.get_layer('reconstructed'))
    # for i in range(7, 10):
    #     print(vae.layers[i].name)
    #     decoder.add(vae.layers[i])
    
    return vae, encoder, decoder

def AE(input_shape, latent_dim=32, intermediate_dim=256, weights=None):
    inputs = layers.Input(shape=input_shape)
    # flattened = layers.Flatten()(inputs)
    # flattened = layers.Dropout(0.3)(flattened)
    flattened = layers.GlobalAveragePooling2D(name='gap')(inputs)
    h = layers.Dense(intermediate_dim, name='intermediate_dense_encoder_1', activation='tanh')(flattened)
    h = layers.Dense(intermediate_dim, name='intermediate_dense_encoder_2', activation='tanh')(h)
    z = layers.Dense(latent_dim, name='z')(h)
    
    h = layers.Dense(intermediate_dim, input_shape=input_shape, name='intermediate_dense_decoder_1', activation='linear')(z)
    h = layers.Dense(intermediate_dim, input_shape=input_shape, name='intermediate_dense_decoder_2', activation='linear')(h)
    h = layers.Dense(tf.reduce_prod(input_shape), activation='linear', name='expand')(h)
    reconstructed = layers.Reshape(input_shape, name='restore_shape')(h)
    ae = models.Model(inputs, reconstructed)
    
    # load pretrained weights if specified
    if weights is not None:
        ae.load_weights(weights)
        
    encoder = models.Model(ae.inputs, ae.get_layer('z').output)
    
    decoder = tf.keras.models.Sequential()
    decoder.add(tf.keras.layers.Input(shape=latent_dim,))
    decoder.add(ae.get_layer('intermediate_dense_decoder_1'))
    decoder.add(ae.get_layer('intermediate_dense_decoder_2'))
    decoder.add(ae.get_layer('expand'))
    decoder.add(ae.get_layer('restore_shape'))
    
    return ae, encoder, decoder