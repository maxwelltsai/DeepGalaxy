from data_io_new import DataIO
from models import *
import numpy as np
import h5py

backbone = get_base_model('EfficientNetB0', weights='imagenet', include_top=False, input_shape=(512,512,3))
# model, encoder, decoder = FCAE(backbone_name='EfficientNetB0', input_shape=(512,512,1))
model, encoder, decoder = FCAE(backbone=backbone, input_shape=(512,512,1))
model.load_weights('features_model.h5')
vae, ve, vd = VAE(input_shape=encoder.output_shape[1:])
# vae, ve, vd = AE(input_shape=encoder.output_shape[1:])

data_io = DataIO()
X, Y, num_classes = data_io.load_partial('../output_bw_512.hdf5', dset_name_pattern='s_*', camera_pos='*')
# X, Y, num_classes = data_io.load_partial('../output_bw_512.hdf5', dset_name_pattern='s_1_m_1*', camera_pos=[1,2,3])
train_images = X.astype(np.float32)

features = encoder.predict(train_images, verbose=1)
print(features.max(), features.min())
# features_normed = features / np.max(features)
# vae.compile(optimizer='adam', loss='binary_crossentropy')
# vae.compile(optimizer='adam', loss='mse')
vae.compile(optimizer='adam', loss=nll)
print(vae.summary())
vae.fit(features,features, epochs=10, batch_size=32)

vae.save_weights('fvae.h5')

# model, encoder, decoder = FCAE(backbone_name='EfficientNetB0', input_shape=(512,512,1), weights='features_model.h5')
z_train_mu, z_train_log_var, z_train = ve.predict(features, verbose=1)
with h5py.File('z_train.h5', 'w') as h5f:
    h5f.create_dataset('z_train_mu', data=z_train_mu)
    h5f.create_dataset('z_train_log_var', data=z_train_log_var)
    h5f.create_dataset('z_train', data=z_train)

# z = ve.predict(features, verbose=1)
# with h5py.File('z_train.h5', 'w') as h5f:
#     h5f.create_dataset('z', data=z)
