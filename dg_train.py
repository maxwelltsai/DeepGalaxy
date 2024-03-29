"""
Deep Galaxy Training code.

Maxwell Cai (SURF), October 2019 - May 2020.

# Dynamically creating training sessions with commandline arguments.
"""

import argparse
from deep_galaxy_training import DeepGalaxyTraining


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--arch', dest='dnn_arch', type=str, default='EfficientNetB7', help='CNN architecture (e.g., EfficientNetB7, ResNet50, VGG16), case-sensitive')
    # parser.add_argument('--weights', dest='dnn_weights', type=str, default='imagenet', help='Use pretrained weights as an initialization (e.g., imagenet)')
    parser.add_argument('-f', '--file', dest='file_name', type=str, required=True, help='File name of the DeepGalaxy HDF5 dataset')
    parser.add_argument('-d', '--datasets', dest='datasets', type=str, default='s_1_m_1*', help='Name pattern of the selected datasets (supports regexp)')
    parser.add_argument('-o', '--optimizer', dest='optimizer', type=str, default='Adadelta', help='Optimizer for the gradient descent')
    parser.add_argument('-l', '--learning-rate', dest='lr', type=float, default=1, help='Learning rate (depends on the optimizer)')
    parser.add_argument('-m', '--data-loading-mode', dest='data_loading_mode', type=int, default=0, help='Data loading model. -1: load the full dataset of each compute node; 0: each node loads a fraction of the full dataste according to the MPI rank and size; >0: each node loads a fraction of the full dataset, and reload+shuffle the fraction per every -m epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=4, help='batch size (per processor)')
    parser.add_argument("--multi-gpu", type=bool, nargs='?', const=True, default=False, help="Use Keras multi_gpu API (depreciated)")
    parser.add_argument('--distributed', dest='distributed', action='store_true', default=True, help='Turn on Horovod distributed training')
    parser.add_argument('--allow-growth', dest='allow_growth', action='store_true', default=True, help='Allow GPU memory to grow dypnamically according to the size of the model.')
    parser.add_argument('--debug', dest='debug_mode', action='store_true', default=False, help='Enable debug model (invokes TensorBoard callback, generates TF timeline, etc.)')
    parser.add_argument('--gpu-mem-frac', dest='gpu_mem_frac', type=float, default=None, help='Fraction of GPU memory to allocate per process. If None, this is handled automaticaly. If a number > 1 is given, unified memory is used.')
    parser.add_argument('--no-distributed', dest='distributed', action='store_false', help='Turn off Horovid distributed training')
    parser.add_argument('--noise', dest='noise_stddev', type=float, default=0.08, help='The stddev of the Gaussian noise for mitigatyying overfitting')
    parser.add_argument('--num-camera', dest='num_cam', type=int, default=14, help='Number of camera positions (for data augmentation). Choose an integer between 1 and 14')
    parser.add_argument('--weights', dest='weights', type=str, default=None, help='Weights for initialization. Could be imagenet or None.')
    args = parser.parse_args()

    print(args)

    dgtrain = DeepGalaxyTraining()
    dgtrain.distributed_training = args.distributed
    dgtrain.multi_gpu_training = args.multi_gpu
    dgtrain.base_model_name = args.dnn_arch
    dgtrain.noise_stddev = args.noise_stddev
    dgtrain.batch_size = args.batch_size
    dgtrain.learning_rate = args.lr
    dgtrain.epochs = args.epochs
    dgtrain.debug_mode = args.debug_mode
    dgtrain.data_loading_mode = args.data_loading_mode
    dgtrain._gpu_memory_allow_growth = args.allow_growth
    dgtrain.weights_init = args.weights
    if args.gpu_mem_frac is None:
        dgtrain._gpu_memory_fraction = None
    else:
        dgtrain._gpu_memory_fraction = float(args.gpu_mem_frac)
    dgtrain.initialize()
    dgtrain.load_data(args.file_name, dset_name_pattern=args.datasets, camera_pos=range(0, args.num_cam))
    dgtrain.load_model()
    dgtrain.fit()
    dgtrain.save_model()
    dgtrain.finalize()
