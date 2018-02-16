# Parle: parallelizing stochastic gradient descent

This is the code for [Parle: parallelizing stochastic gradient descent](https://arxiv.org/abs/1707.00424). We demonstrate an algorithm for parallel training of deep neural networks which trains multiple copies of the same network in parallel, called as "replicas", with special coupling upon their weights to obtain significantly improved generalization performance over a single network as well as 2-5x faster convergence over a data-parallel implementation of SGD for a single network.

### High-performance multi-GPU version coming soon.

<p align="center">
<img src="https://i.imgur.com/KZlZ3Nw.jpg" width="350">
</p>

We have two versions, both of which are written using [PyTorch](http://pytorch.org):

- A parallel version that uses MPI (mpi4py) for synchronizing weights.
- A more efficient version that can be executed on a single computer with multiple GPUs. The synchronization of weights is done explicitly here using inter-GPU messages.

In both cases, we construct an optimizer class that initializes the requisite buffers on different GPUs and handles all the updates after each mini-batch. As an example, we have provided code for MNIST and CIFAR-10 datasets with two prototypical networks, LeNet and [All-CNN](https://arxiv.org/abs/1412.6806), respectively. The MNIST and CIFAR-10/100 datasets will be downloaded and pre-processed (stored in the ``proc`` folder) the first time ``parle`` is run.

### Instructions for running the code

The MPI version works great for small experiments and prototyping while the second version is a good alternative for larger networks, e.g., [wide-residual networks](https://arxiv.org/abs/1605.07146) used in the paper.

Parle is **very insensitive to hyper-parameters**. A description for some of the parameters and their intuition follows.
- the learning rate ``lr`` is set to be the same as SGD, along with the same drop schedule. It is advisable to train with SGD for a few epochs and then use the same ``lr`` for Parle.
- ``gamma`` controls how far successive gradient updates on each replica are allowed to go from the previous checkpoint, i.e., the last instant when weights were synchronized with the master. This is the same as the step-size in proximal point iteration.
- ``rho`` controls how far each replica moves from the master. The weights of the master are the average of the weights of all the replicas while each replica gets pulled towards this average with a force that is proportional to ``rho``.
- ``L`` is the number of gradient updates performed on each replica (worker) before synchronizing the weights with the master. You can safely fix this to 25. Alternatively, you set this to ``L = gamma x lr`` which has the advantage of being slightly faster towards the end of training.
Proximal point iteration is insensitive to both ``gamma`` and ``rho`` and the above code uses a default decaying schedules for these, which should typically work. In particular, we set ``gamma = rho = 100*(1-/(2 nb)^(k/L)`` where ``nb`` is the number of mini-batches per epoch and ``k`` is the current iteration number. ``L`` is the number of weight updates per synchronization, as above.
- ``n`` is the number of replicas. The code distributes these replicas on all available GPUs. For the MPI version, this is controlled by ``MPI.RANK``. In general, larger the ``n``, the better Parle works. Each replica can itself be data-parallel using multiple GPUs.

The number of epochs ``B`` for Parle is typically much smaller than SGD and 5-10 epochs are sufficient to train on MNIST or CIFAR-10/100.

1. Execute ``python parle_mpi.py -h`` to get a list of all arguments and defaults. You can train LeNet on MNIST with 3 replicas using
    ```
    python parle_mpi.py -n 3
    ```
2. You can train All-CNN on CIFAR-10 with 3 replicas using
    ```
    python parle_mpi.py -n 3 -m allcnn
    ```
3. You can run the MPI version with 12 replicas as

    ```
    mpirun -n 12 python parle.py
    ```

### Special cases
1. Setting ``n=1, L=1, gamma=0, rho=0`` makes Parle equivalent to SGD; the implementation here uses Nesterov's momentum.
2. Setting ``n=1, rho=0`` decouples the replicas from the master. In this case, Parle becomes equivalent to executing [Entropy-SGD: biasing gradient descent into wide valleys](https://arxiv.org/abs/1611.01838); see the code for the latter [here](https://github.com/ucla-vision/entropy-sgd).
3. Setting ``L=1, gamma=0`` makes Parle equivalent to [Elastic-SGD](https://arxiv.org/abs/1412.6651); the code for the latter by the original authors is [here](https://github.com/sixin-zh/mpiT). Parle uses an annealing schedule on ``rho`` however, which makes it faster and generalize better than vanilla Elastic-SGD.
