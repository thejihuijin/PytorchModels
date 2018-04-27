import os
import sys
import time
import numpy
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams
import pickle
import cPickle
import math
import time
import scipy.io as sio
import scipy.misc



def relu(x):
    return theano.tensor.switch(x < 0., 0., x)

# Convolutional Layer Class
class LeNetConvPoolLayer(object):


    def __init__(self, rng, input, filter_shape, image_shape, W, b, type_conv='full', poolsize=(1,1)):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # initialize weights with random weights
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))

        if W is None:

            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        # initialize biases with random weights
        if b is None:

            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)

        self.W = W
        self.b = b

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode = type_conv)

        self.output = relu((conv_out + self.b.dimshuffle('x', 0, 'x', 'x')))
        # store parameters of this layer
        self.params = [self.W, self.b]


def DeepInverse(batch_size = 100):



    rng = numpy.random.RandomState(123)

    ## Loading saved weights and biases / if any
    # To be filled by user
    f = open('address/to/the/saved/network', 'U')
    models = cPickle.load(f)
    f.close()

    ## Loading training and test data. Data should be a 2D array where each row corresponds to an image.
    ## We reshape it in the next line. train_set_x contains signal/image proxies and train_set_y_ contains
    ## real signal/image

    train_set_x = numpy.load('address of numpy array containing training data')
    train_set_y = numpy.load('address of numpy array containing testing data')

    test_set_x = numpy.load('address of numpy array containing training data')
    test_set_y = numpy.load('address of numpy array containing testing data')

    train_set_x = theano.shared(numpy.asarray(train_set_x,dtype=theano.config.floatX),borrow=True)
    train_set_y = theano.shared(numpy.asarray(train_set_y,dtype=theano.config.floatX),borrow=True)


    test_set_x = theano.shared(numpy.asarray(test_set_x,dtype=theano.config.floatX),borrow=True)
    test_set_y = theano.shared(numpy.asarray(test_set_y,dtype=theano.config.floatX),borrow=True)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_test_batches /= batch_size


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # signal/image proxy
    y = T.matrix('y')  # output


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape input and output batch to (batch_size,64*64)

    layer1_input = x.reshape(( batch_size, 1, 64, 64))
    correct_output = y.reshape(( batch_size, 1, 64, 64))


    # First Layer:
    # Type: Convolutional Layer

    layer1 = LeNetConvPoolLayer(rng, input= layer1_input,
        image_shape=(batch_size, 1, 64, 64),
        filter_shape=(64, 1,11,11), type_conv = 'full', poolsize = (1,1))
    layer1_output = layer1.output[:,:,5:-5,5:-5]


    # Second Layer:
    # Type: Convolutional Layer

    layer2 = LeNetConvPoolLayer(rng, input= layer1_output,
            image_shape=(batch_size, 64, 64, 64),
            filter_shape=(32, 64, 11, 11), type_conv = 'full', poolsize = (1,1))

    layer2_output = layer2.output[:,:,5:-5,5:-5]

    # Third Layer:
    # Type: Convolutional Layer

    layer3 = LeNetConvPoolLayer(rng, input= layer2_output,
            image_shape=(batch_size, 32, 64, 64),
            filter_shape=(1, 32, 11, 11), type_conv = 'full', poolsize = (1,1))

    layer3_output = layer3.output[:,:,5:-5,5:-5]

    # the cost we minimize during training is the MSE of the model
    cost = T.sum(T.sqr(layer3_output - correct_output))/batch_size


    # What we are doing in the Testing Phase
    test_model = theano.function([index], cost,
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})


    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)


   # SGD for updating weights and biases
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    n_epochs = 1000
    validation_frequency = n_train_batches/10

    best_params = None
    best_MSE = 1000000000
    best_iter = 0

    epoch = 0

    while (epoch < n_epochs):

        epoch = epoch + 1

        for minibatch_index in xrange(n_train_batches):

            # Calculating Cost function
            iter = (epoch - 1) * n_train_batches + minibatch_index
            cost_ij = train_model(minibatch_index)
            print (cost_ij)


            if (iter + 1) % validation_frequency == 0:

                # test it on the test set

                test_losses = [test_model(i) for i in xrange(n_test_batches)]
                test_score = numpy.mean(test_losses)

                print(('     epoch %i, minibatch %i/%i, Average MSE is '
                       ' %f ') %
                      (epoch, minibatch_index + 1, n_train_batches,
                       test_score))

                # Saving the parameters if outperforming previously saved networks

                if (test_score < best_MSE):

                    best_MSE = test_score
                    best_iter = iter
                    best_params = params


    print('Optimization complete.')
    print('Best Average PSNR was %f obtained at iteration %i,' %
          (best_MSE , best_iter + 1))


    # To be filled with the user
    f = open('address/to/save/the/file.pkl', 'wb')
    pickle.dump(best_params, f)
    f.close()


if __name__ == '__main__':
    evaluate_DeepInverse()


