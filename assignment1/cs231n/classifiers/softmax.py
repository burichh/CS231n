from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]
    
    o = np.zeros(shape=(N, C))
    s = np.zeros(shape=(N, C))

    for n in range(N):
        for j in range(C):
            for k in range(D):
                o[n, j] += X[n,k] * W[k,j]

    o_max = np.reshape(np.max(o, axis=1), (-1, 1))
    exp_shifted = np.exp(o - o_max)
    sum = np.sum(exp_shifted, axis=1)
    s = exp_shifted / np.reshape(sum, (-1,1))

    loss = np.average(o_max - o[range(N), y] + np.log(sum)) + reg * np.sum(np.power(W, 2))

    for i in range(D):
        for j in range(C):
            for n in range(N):
                dW[i, j] += X[n, i] * s[n, j]
                if(j == y[n]):
                    dW[i, j] -= X[n, i]

    dW /= N
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    o = np.matmul(X,W)
    o_correct = o[range(N), y].reshape(-1,1)
    o_max = np.max(o, axis=1, keepdims=True)
    exp_stable = np.exp(o - o_max)
    sum = np.sum(exp_stable, axis=1, keepdims=True)

    loss = np.average(o_max - o_correct + np.log(sum)) + reg * np.sum(np.power(W, 2))

    s = exp_stable / sum
    s[range(N), y] -= 1

    dW = np.matmul(np.transpose(X), s) / N + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
