# CNN MNIST Siamese network

This example shows a simple example of the DAG interface for a MNIST Siamese network. This network is a simple extension of the original MNIST network with two branches connected to a Contrastive loss [[1]](#hadsell16).

The training is done on the original MNIST data and learns an embedding to a 2D space which is simple to visualize.

## Install module
The simplest way how to install the module is to run:
```matlab
vl_contrib install siamese-mnist
vl_contrib setup siamese-mnist
edit siamese_mnist_example
```
## Example Code
An example tutorial on how to run the training and evaluate a [MNIST Siamese network](net.png) is in the MATLAB Live Script `example.mlx`. It shows how to train and cut the Siamese network to parts in order to obtain the final feature embedding.

The final embedding should look similar to this: ![Final embedding of the Siamese training](embedding.png)

## Contents of the package
* `cnn_mnist_siames.m` Network and training definition script.
* `vl_create_siamese.m` Helper function to create a Siamese network.
* `vl_nncontrloss.m` Implementation of the Contrastive loss.


<a name="hadsell06"></a>
[1] Hadsell, Raia, Sumit Chopra, and Yann LeCun.
"Dimensionality reduction by learning an invariant mapping"
In Proc. of CVPR 2006.
