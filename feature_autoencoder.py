from data_io_new import DataIO
from models import *
import numpy as np

backbone = get_base_model('VGG16', weights='imagenet', include_top=False, input_shape=(512,512,3))
# model, encoder, decoder = FCAE(backbone_name='EfficientNetB0', input_shape=(512,512,1))
model, encoder, decoder = FCAE(backbone=backbone, input_shape=(512,512,1))

data_io = DataIO()
# X, Y = data_io.load_partial('../output_bw_512.hdf5', dset_name_pattern='s_*', camera_pos=[4,5,6])
X, Y, num_classes = data_io.load_partial('../output_bw_512.hdf5', dset_name_pattern='s_*', camera_pos='*')
# X, Y, num_classes = data_io.load_partial('../output_bw_512.hdf5', dset_name_pattern='s_1_m_1*', camera_pos='*')
# train_images = X.astype(np.float32)
train_images = X 

model.compile(optimizer='adam', loss='mae')
print(encoder.summary())
print(model.summary())

model.fit(train_images, train_images, epochs=5, batch_size=8)

model.save_weights('features_model.h5')