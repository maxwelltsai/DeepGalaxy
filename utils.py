import matplotlib.pyplot as plt
import numpy as np 
import cv2 
import glob 
import os 
import tensorflow as tf 


def plot_latent_images(model, n, digit_size=16):
    """Plots n x n digit images decoded from the latent space."""

#     norm = tfp.distributions.Normal(0, 1)
    norm = np.random.normal(0, 1)
    grid_x = np.linspace(-30, 30, n)
    grid_y = np.linspace(-30, 30, n)
#     grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
#     grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
    feature_recon_width = digit_size*n
    feature_recon_height = feature_recon_width
    feature_recon = np.zeros((feature_recon_width, feature_recon_height))
    image_recon_width = 512*n
    image_recon_height = image_recon_width
    image_recon = np.zeros((image_recon_width, image_recon_height))
    latent_dim = 8

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.ones((1, latent_dim))
            z[0,0] = xi
            z[0,1] = yi
#             z[0,2] = xi
#             z[0,3] = yi
        
            z_decoded = vae_decoder(z)
            image = decoder(z_decoded)  

            feature = tf.reshape(tf.reduce_sum(z_decoded, axis=-1), (digit_size, digit_size))
            feature_recon[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = feature.numpy()
    
            image = tf.reshape(image, (512, 512))
            image_recon[i * 512: (i + 1) * 512, j * 512: (j + 1) * 512] = image.numpy()
  
    plt.figure(0, figsize=(20, 20))
    plt.imshow(feature_recon, cmap='Greys_r')
    plt.axis('Off')
    plt.title('feature')
    plt.show()

    plt.figure(1, figsize=(20, 20))
    plt.imshow(image_recon, cmap='Greys_r')
    plt.axis('Off')
    plt.show()
    
def find_similar_images(feature_encoder, vae_encoder, train_images, z_train, query_img_array, k=5):
    encoded_features = feature_encoder.predict(query_img_array)
    z_mu_features, z_log_var_features, z_features = vae_encoder.predict(encoded_features)
    z_representation = np.concatenate([z_mu_features, z_log_var_features], axis=-1)
    print(z_representation.shape)
    for query_img_id in range(len(query_img_array)):
        print(query_img_id)
        z_dist = tf.norm(z_representation[query_img_id] - z_train, axis=-1)
        z_dist_unique, z_dist_unique_id = np.unique(z_dist, return_index=True)
        panel_size = 12
        
        plt.figure(11, figsize=(panel_size, panel_size*(k+1)))
        ax = plt.subplot(1, k+1, 1)
        ax.imshow(query_img_array[query_img_id][:, :, 0].astype(np.float32))
        ax.set_title('obs')
        # ax = plt.subplot(1, 2, 2)
        # ax.imshow(decoded_image[0, :, :, 0])
        # ax.set_title('reconstructed')

        for i in range(2, k+2):
            ax = plt.subplot(1, k+1, i)
            ax.imshow(train_images[z_dist_unique_id[i-2]][:, :, 0].astype(np.float32))
            ax.set_title('%d, %.2f' % (z_dist_unique_id[i-2], z_dist_unique[i-2]))
        plt.show()
        plt.tight_layout()
        
    
    
def generate_similar_images(feature_encoder, feature_decoder, vae_encoder, vae_decoder, z_train, query_img_array, k=5):
    encoded_features = feature_encoder.predict(query_img_array)
    z_mu_features, z_log_var_features, z_features = vae_encoder.predict(encoded_features)
    for query_img_id in range(len(query_img_array)):
        z_dist = tf.norm(z_features[query_img_id] - z_train, axis=-1)
        z_dist_unique, z_dist_unique_id = np.unique(z_dist, return_index=True)
        reconstructed_features = vae_decoder.predict(z_train[z_dist_unique_id[:k]])
        decoded_images = feature_decoder.predict(reconstructed_features)
        print(decoded_images.shape)
        panel_size = 12
        
        plt.figure(11, figsize=(panel_size, panel_size*(k+1)))
        ax = plt.subplot(1, k+1, 1)
        ax.imshow(query_img_array[query_img_id][:, :, 0])
        ax.set_title('obs')
        
        # ax = plt.subplot(1, 2, 2)
        # ax.imshow(decoded_image[0, :, :, 0])
        # ax.set_title('reconstructed')

        for i in range(2, k+1):
            ax = plt.subplot(1, k+1, i)
            ax.imshow(decoded_images[i-2][:, :, 0])
            ax.set_title('%d, %.2f' % (z_dist_unique_id[i-2], z_dist_unique[i-2]))
        plt.show()
        plt.tight_layout()
    
def generate_z_train(train_images, feature_encoder, vae_encoder):
    """
    Generate a numpy array containing the latent-space representation of the training images.
    """
    features = feature_encoder.predict(train_images, verbose=1)
    z_train = vae_encoder.predict(features, verbose=1)
    return z_train
    
def load_images_from_directory(directory, postfix='jpg', img_size=512):
    img_list = glob.glob(os.path.join(directory, '*.%s' % postfix))
    print(img_list)

    obs_imgs = []
    obs_imgs_title = []
    for img_name in img_list:
        im = cv2.imread(img_name)
        im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_resized = cv2.resize(im_grey, (img_size,img_size))
        print(img_name, im_resized.shape)
        obs_imgs.append((im_resized.reshape(img_size,img_size,1)/255).astype(np.float32))
        obs_imgs_title.append(os.path.basename(img_name))
    print(obs_imgs_title)
    obs_imgs = np.array(obs_imgs)
    return obs_imgs