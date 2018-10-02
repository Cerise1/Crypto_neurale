# Adversarial Neural Cryptography in [TensorFlow](https://github.com/tensorflow/tensorflow)

A Tensorflow Flow implementation of Google Brain's recent paper ([Learning to Protect Communications with Adversarial Neural Cryptography.](https://arxiv.org/pdf/1610.06918v1.pdf))

Two Neural Networks, Alice and Bob learn to communicate secretly with each other, in presence of an adversary Eve.
Some optimisation from original paper have been done so that Eve converge towards true random and the ciphertext created can be a binary vector.


![Setup](assets/diagram.png)

## Pre-requisites

* TensorFlow 
* Matplotlib
* Numpy

## Usage 
First, ensure you have the dependencies installed.

    $ pip install -r requirements.txt

To train the neural networks, run the `main.py` script.

    $ python main.py --msg-len 16 --epochs 10000
    
    
## Attribution / Thanks

* carpedm20's DCGAN [implementation](https://github.com/carpedm20/DCGAN-tensorflow) in TensorFlow. 
* Liam's [implementation](https://github.com/nlml/adversarial-neural-crypt) of Adversarial Neural Cryptography in Theano. 
