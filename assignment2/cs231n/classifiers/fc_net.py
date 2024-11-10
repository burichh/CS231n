from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt 

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dimensions = [input_dim] + hidden_dims + [num_classes]

        for i in range(1, len(dimensions)):
            self.params['W' + str(i)] = weight_scale * np.random.randn(dimensions[i-1], dimensions[i])
            self.params['b' + str(i)] = np.zeros(dimensions[i])

        if self.normalization == "batchnorm" or self.normalization == "layernorm":
            for i in range(1, len(dimensions) - 1):
                self.params["gamma" + str(i)] = np.ones(dimensions[i])
                self.params["beta" + str(i)] = np.zeros(dimensions[i])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        cache = {}

        for i in range(1, self.num_layers):
            w = self.params['W' + str(i)]
            b = self.params['b' + str(i)]

            if self.normalization == "batchnorm":
                gamma = self.params['gamma' + str(i)]
                beta = self.params['beta' + str(i)]
                X, cache[str(i)] = affine_bn_relu_forward(X, w, b, gamma, beta, self.bn_params[i-1])
            elif self.normalization == "layernorm":
                gamma = self.params['gamma' + str(i)]
                beta = self.params['beta' + str(i)]
                X, cache[str(i)] = affine_ln_relu_forward(X, w, b, gamma, beta, self.bn_params[i-1])
            else:
                X, cache[str(i)] = affine_relu_forward(X, w, b)

            if self.use_dropout:
                X, cache[str(i) + "_dropout"] = dropout_forward(X, self.dropout_param)

        w = self.params['W' + str(self.num_layers)]
        b = self.params['b' + str(self.num_layers)]
        scores, score_cache = affine_forward(X, w, b)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        reg_loss = 0
        for i in range(self.num_layers):
            reg_loss += np.sum(np.power(self.params['W' + str(i+1)], 2))
        reg_loss *= (0.5 * self.reg)

        loss, dscores = softmax_loss(scores, y)
        loss += reg_loss

        dx, dw, db = affine_backward(dscores, score_cache)
        grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db

        for i in range(self.num_layers-1, 0, -1):
            if self.use_dropout:
                dx = dropout_backward(dx, cache[str(i) + "_dropout"])

            if self.normalization == "batchnorm":
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dx, cache[str(i)])
                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta
            elif self.normalization == "layernorm":
                dx, dw, db, dgamma, dbeta = affine_ln_relu_backward(dx, cache[str(i)])
                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta
            else:
                dx, dw, db = affine_relu_backward(dx, cache[str(i)])
            
            grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = db

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    
    def loss_with_plot(self, X, example_idx=0, y=None):
        cache = {}

        fig, axs = plt.subplots(2, self.num_layers + 1, figsize=(12, 4))

        activation_hist_range = None#(0,100)
        activation_bin_num = 30

        axs[0, 0].hist(X[example_idx].reshape(-1,), bins=10, color='blue')

        for i in range(1, self.num_layers):
            w = self.params['W' + str(i)]
            b = self.params['b' + str(i)]
            X, cache[str(i)] = affine_relu_forward(X, w, b)
            axs[0, i].hist(X[example_idx], bins=activation_bin_num, color='blue', range=activation_hist_range)

        w = self.params['W' + str(self.num_layers)]
        b = self.params['b' + str(self.num_layers)]
        scores, score_cache = affine_forward(X, w, b)

        axs[0, self.num_layers].hist(scores[example_idx], bins=activation_bin_num, color='blue', range=activation_hist_range)

        np.set_printoptions(precision=2)
        print(scores[example_idx])
        print(np.argmax(scores[example_idx]))

        mode = "test" if y is None else "train"
        if mode == "test":
            return

        loss, grads = 0.0, {}

        reg_loss = 0
        for i in range(self.num_layers):
            reg_loss += np.sum(np.power(self.params['W' + str(i+1)], 2))
        reg_loss *= (0.5 * self.reg)

        loss, dscores = softmax_loss(scores, y)
        loss += reg_loss

        grad_hist_range_max = 2.0
        grad_hist_range = None#(-grad_hist_range_max, grad_hist_range_max)

        dx, dw, db = affine_backward(dscores, score_cache)
        grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db
        axs[1, self.num_layers].hist(dw.reshape(-1,), bins=30, color='red', range=grad_hist_range)

        for i in range(self.num_layers-1, 0, -1):
            dx, dw, db = affine_relu_backward(dx, cache[str(i)])
            grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = db
            axs[1, i].hist(dw.reshape(-1,), bins=30, color='red', range=grad_hist_range)