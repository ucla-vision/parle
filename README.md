# Parle: parallel training of deep neural networks

This is the code for [Parle: parallel training of deep neural networks](https://arxiv.org/abs/1707.00424). We demonstrate an algorithm for parallel training of deep neural networks which trains multiple copies of the same network in parallel, called as "replicas", with special coupling upon their weights to obtain significantly improved generalization performance over a single network as well as 2-4x faster convergence over a data-parallel implementation of SGD for a single network.

<p align="center">
<img src="https://i.imgur.com/KZlZ3Nw.jpg" width="350">
</p>

We have two versions, both of which are written using [PyTorch](http://pytorch.org):

- A parallel version that uses MPI (mpi4py) for synchronizing weights.
- A more efficient version that can be executed on a single computer with multiple GPUs. The synchronization of weights is done explicitly here using inter-GPU messages.

In both cases, we construct an optimizer class that initializes the requisite buffers on different GPUs and handles all the updates after each mini-batch. As an example, we have provided code for MNIST and CIFAR-10 datasets with two prototypical networks, LeNet and [All-CNN](https://arxiv.org/abs/1412.6806), respectively. The MNIST and CIFAR-10/100 datasets will be downloaded and pre-processed (stored in the ``proc`` folder) the first time ``parle`` is run.

### Instructions for running the code

The MPI version works great for small experiments and prototyping while the second version is a good alternative for larger networks, e.g., [wide-residual networks](https://arxiv.org/abs/1605.07146) used in the paper. The various parameters for Parle are:
- ``L`` is the number of gradient updates performed on each replica before synchronizing the weights with the master
- ``gamma`` controls how far successive gradient updates on each replica are allowed to go from the previous checkpoint, i.e., the last instant when weights were synchronized with the master. This is the same as the step-size in proximal gradient descent algorithms.
- ``rho`` controls how far each replica moves from the master. The weights of the master are the average of the weights of all the replicas while each replica gets pulled towards this average with a force that is proportional to ``rho``.
- ``n`` is the number of replicas. The code distributes these replicas on all available GPUs. For the MPI version, this is controlled by ``MPI.RANK``.

The algorithm is very insensitive to the values of ``L, gamma, rho`` and the only important hyper-parameter is the learning rate ``lr``. It is advisable to set ``lr`` to be the same as SGD. The number of epochs ``B`` for Parle is typically much smaller than SGD and 5-10 epochs are sufficient to train on MNIST or CIFAR-10/100.

1. Execute ``python parle_mpi.py -h`` to get a list of all arguments and defaults. You can train LeNet on MNIST with 3 replicas using
    ```
    python parle_mpi.py -n 3
    ```
2. Execute ``python parle.py -h`` to get a list of all arguments and defaults. You can train All-CNN on CIFAR-10 with 3 replicas using
    ```
    python parle.py -n 3 -m allcnn
    ```

### Special cases
1. Setting ``n=1, L=1, gamma=0, rho=0`` makes Parle equivalent to SGD; the implementation here uses Nesterov's momentum.
2. Setting ``n=1, rho=0`` decouples the replicas from the master. In this case, Parle becomes equivalent to executing [Entropy-SGD: biasing gradient descent into wide valleys](https://arxiv.org/abs/1611.01838); see the code for the latter [here](https://github.com/ucla-vision/entropy-sgd).
3. Setting ``L=1, gamma=0`` makes Parle equivalent to [Elastic-SGD](https://arxiv.org/abs/1412.6651); the code for the latter by the original authors is [here](https://github.com/sixin-zh/mpiT).