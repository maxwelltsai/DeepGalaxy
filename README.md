di# DeepGalaxy: Classifying the properties of galaxy mergers using Deep Convolutional Networks

This project aims to leverage the capability of pattern recognization in modern deep learning to classify the properties of galaxy mergers. Galaxy mergers (the dynamical process during which two galaxies collide) are among the most spectacular phenomena in the Universe. During the merger process, the two interacting galaxies are tidally distorted, producing elongated shaped galaxies with tidal arms. Many irregular galaxies are created through this mechanism, and the evolving visual feature in this dynamic process can be considered as an image classification problem in the fields of computer vision and machine learning.

In this project, a deep convolutional neural network (CNN) is trained with visualization of galaxy merger simulations. Accordingly, the dynamical properties (e.g., the timescale in which two galaxies collide) are obtained from the numerical simulations, which are then encoded as the labels (in the supervised learning process). 

The simulations are carried out using `Bonsai` (Bédorf et al., 2012, 2019), a GPU-accelerated Barnes-Hut tree code. GitHub: https://github.com/treecode/Bonsai. 

The CNN is built with state-of-the-art architectures, such as [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946) and [ResNet (He et al. 2019)](https://arxiv.org/abs/1512.03385). The `EfficientNet` implementation is based on [this repository](https://github.com/qubvel/efficientnet). The implementations of other CNNs (including `ResNet50`) are based on the `tf.keras.applications` module. A full list of supported CNNs can be found at https://www.tensorflow.org/api_docs/python/tf/keras/applications.




## Prerequisites
- TensorFlow (1.14 or above, 2.1 or above)
- Scikit-learn
- Scikit-image
- h5py

It is recommended to install these package in a Python virtual environment.

## Training data
We simulated galaxy mergers of different mass ratios and size ratios (1:1, 1:2, 1:3, 2:3). The snapshots of the simulations are visualized once every 5 Myr (million years) using cameras from 14 different positions, and therefore generating 14 2D images. Each simulation should complete within a timescale of about 700 Myr. These images are stored in a compressed HDF5 dataset. The available image resolution are (256, 256), (512, 512), (1024, 1024), and (2048, 2048) pixels. 

The image dataset of (512, 512) pixels can be downloaded from here: https://surfdrive.surf.nl/files/index.php/s/Mzm28FQ1udG3FG7 (2GB)


## Training

The training can be done on a single node or multiple nodes. If you are running the code on an HPC cluster, please allocate resources prior to running the code. The file `dg_train.py` is the training script to run. It supports a number of command-line arguments. To obtain a full list of arguments, type `python dg_train.py -h`. Here are some useful ones:

- `-f <dataset.hdf5>`: the file name and path of the HDF5 image dataset (required).
- `--epochs`: The number of epoches to carry out the training. 
- `--arch`: the CNN architecture to use, e.g., `EfficientNetB4`, `EfficientNetB7`, `ResNet50`.
- `--batch-size`: the batch size to use in the training. Usually this is constrained by the size of (GPU) memory.
- `--num-camera`: the number of camera positions to use for data augmentation. An integer from 1 to 14 is acceptable.
- `-d`: datasets to use for training. To use all datasets stored in the HDF5 file, type `s_*`. To include datasets only with size ratio of 1.5, type `s_1.5_*`, and so on.
- `--noise`: mitigating overfitting by imposing a random Gaussian noise. This argument specifies the standard derivation of the noise. 


### Training on a single node.

```
python dg_train.py -f output_bw_512.hdf5 --epochs 20 --noise 0.3
```
Please replace `output_bw_512.hdf5` with the actual file name of the HDF5 dataset. Change the epochs and noise, and other parameters whenever necessary.

### Training on multiple nodes.

Multi-node training relies on [`horovod`](https://github.com/horovod/horovod), a distributed training framework built on the top of DL frameworks and collective communication protocols. To carry out a parallel training session, the following command can be used in the batch script:

```
mpirun -np 32 python dg_train.py -f output_bw_512.hdf5 --epochs 20 --noise 0.3
```
where `-np 32` should be changed according to the actual number of MPI processes. For example, if 8 nodes are allocated, and each node with 4 processors (e.g., GPU), then `np = 8 * 4 = 32`. If GPUs are used, `DeepGalaxy` will automatically bind each MPI process to a GPU.  Please replace `output_bw_512.hdf5` with the actual file name of the HDF5 dataset. Change the epochs and noise, and other parameters whenever necessary.

## Acknowledgement
This project is supported by [PRACE](https://prace-ri.eu/), [SURF](https://www.surf.nl/en), and [Leiden Observatory](https://www.universiteitleiden.nl/en/science/astronomy).

## Contact
Questions/comments please direct to Maxwell X. Cai: maxwell.cai _at_ surfsara.nl