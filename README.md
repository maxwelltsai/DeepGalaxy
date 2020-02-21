# DeepGalaxy: Classifying the properties of galaxy mergers using EfficientNet

This project aims to leverage the capability of pattern recognization in modern deep learning to classify the properties of galaxy mergers. The neural network is trained with simulated galaxy mergers. The final goal, however, is to use the trained neural network to predict the dynamical properties of observed galaxy mergers (e.g., collision timescale).

The simulations are carried out using `Bonsai` (Bédorf et al., 2012, 2019), a GPU-accelerated Barnes-Hut tree code. GitHub: https://github.com/treecode/Bonsai 

This repository hosts the source code for the model implementation and the training script. The training data is not included in the repository. The training script can be scaled up to multiple nodes, and utilize multiple GPUs (or CPUs) on each node.

To carry out the (parallel) training, the following command can be used in the batch script:

```
mpirun -np 8 python dg_train.py
```
where `-np 8` should be changed according to the actual number of MPI processes. For example, if two nodes are allocated, and each node has 4 GPUs, then `np = 2 * 4 = 8`. Please do not add `srun` before the `mpirun` command, because this will override the environmental variable set in the code according to the `MPI_LOCAL_RANK`, which essentially maps one MPI process to a GPU. If more than one MPI processes are mapped to the same GPU, a `CUDA_OUT_OF_MEMORY_ERROR` will be generated an the training cannot be carried out.

