import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def myprojection(alpha, c=None):
    """ 
    If c=None, then returns the point in [0, infinity]^n closest to alpha.
    If c!=None, then returns the point in [0, c]^n closest to alpha.
    
    """
    with torch.no_grad():
        myhelp = torch.clamp(alpha, 0, max(torch.max(alpha).item(), 0))
        if (c == None):
            return myhelp
        return torch.clamp(myhelp, torch.min(myhelp).item(), c)

def mydualgradient(alpha, kermatrix):
    """ Returns the value at alpha of the gradient of the dual objective w.r.t the Gram matrix kermatrix""" 
    return torch.sub(torch.matmul(kermatrix, alpha), torch.ones(alpha.shape[0], 1))

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw2_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    myret = torch.zeros(x_train.shape[0], 1)
    mykermatrix = torch.FloatTensor([[y_train[i]*y_train[j]*kernel(x_train[i], x_train[j]) for i in range(0, x_train.shape[0])] for j in range(0, x_train.shape[0])])
    for i in range(0, num_iters):
        myret = torch.sub(myret, mydualgradient(myret, mykermatrix), alpha=lr)
        myret = myprojection(myret, c)
    myactualret = torch.FloatTensor([myret[i][0] for i in range(0, myret.shape[0])])
    return myactualret

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw2_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    x_test_predictions = torch.empty(x_test.shape[0])
    for j in range(0, x_test.shape[0]):
        jres = torch.FloatTensor([0])
        for i in range(0, x_train.shape[0]):
            if (alpha[i] == 0):
                continue
            jres = torch.add(jres, alpha[i]*y_train[i]*kernel(x_test[j], x_train[i]))
        x_test_predictions[j] = jres
    return x_test_predictions

class CAFENet(nn.Module):
    mylin = 0
    def __init__(self):
        '''
            Initialize the CAFENet by calling the superclass' constructor
            and initializing a linear layer to use in forward().

            Arguments:
                self: This object.
        '''
        super().__init__(self)
        mylin = nn.Linear(hw2_utils.IMAGE_DIMS[0], hw2_utils.IMAGE_DIMS[1])
        
    def forward(self, x):
        '''
            Computes the network's forward pass on the input tensor.
            Does not apply a softmax or other activation functions.

            Arguments:
                self: This object.
                x: The tensor to compute the forward pass on.
        '''
        pass

def fit(net, X, y, n_epochs=5000):
    '''
    Trains the neural network with CrossEntropyLoss and an Adam optimizer on
    the training set X with training labels Y for n_epochs epochs.

    Arguments:
        net: The neural network to train
        X: n x d tensor
        y: n x 1 tensor
        n_epochs: The number of epochs to train with batch gradient descent.

    Returns:
        List of losses at every epoch, including before training
        (for use in plot_cafe_loss).
    '''
    pass

def plot_cafe_loss():
    '''
    Trains a CAFENet on the CAFE dataset and plots the zero'th through 200'th
    epoch's losses after training. Saves the trained network for use in
    visualize_weights.
    '''
    pass

def visualize_weights():
    '''
    Loads the CAFENet trained in plot_cafe_loss, maps the weights to the grayscale
    range, reshapes the weights into the original CAFE image dimensions, and
    plots the weights, displaying the six weight tensors corresponding to the six
    labels.
    '''
    pass

class DigitsConvNet(nn.Module):
    def __init__(self):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        The layers of your neural network (in order) should be
        1) a 2D convolutional layer with 1 input channel and 8 outputs, with a kernel size of 3, followed by
        2) a 2D maximimum pooling layer, with kernel size 2
        3) a 2D convolutional layer with 8 input channels and 4 output channels, with a kernel size of 3
        4) a fully connected (Linear) layer with 4 inputs and 10 outputs
        '''
        pass

    def forward(self, xb):
        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        pass

def fit_and_evaluate(net, optimizer, loss_func, train, test, n_epochs, batch_size=1):
    '''
    Fits the neural network using the given optimizer, loss function, training set
    Arguments:
        net: the neural network
        optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
        train: a torch.utils.data.Dataset
        test: a torch.utils.data.Dataset
        n_epochs: the number of epochs over which to do gradient descent
        batch_size: the number of samples to use in each batch of gradient descent

    Returns:
        train_epoch_loss, test_epoch_loss: two arrays of length n_epochs+1,
        containing the mean loss at the beginning of training and after each epoch
    '''
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    test_dl = torch.utils.data.DataLoader(test)

    train_losses = []
    test_losses = []

    # Compute the loss on the training and validation sets at the start,
    # being sure not to store gradient information (e.g. with torch.no_grad():)

    # Train the network for n_epochs, storing the training and validation losses
    # after every epoch. Remember not to store gradient information while calling
    # epoch_loss

    return train_losses, test_losses

X, y = hw2_utils.xor_data()
def poly_deg_two(x_test):
    """Linear predictor for XOR data using polynomial kernel of degree 2."""
    alpha = svm_solver(X, y, 0.1, 10000, hw2_utils.poly(2))
    return svm_predictor(alpha, X, y, x_test, hw2_utils.poly(2))

def rbf_sigma_one(x_test):
    """Linear predictor for XOR data using RBF kernel with sigma = 1."""
    alpha = svm_solver(X, y, 0.1, 10000, hw2_utils.rbf(1))
    return svm_predictor(alpha, X, y, x_test, hw2_utils.rbf(1))

def rbf_sigma_two(x_test):
    """Linear predictor for XOR data using RBF kernel with sigma = 2."""
    alpha = svm_solver(X, y, 0.1, 10000, hw2_utils.rbf(2))
    return svm_predictor(alpha, X, y, x_test, hw2_utils.rbf(2))

def rbf_sigma_four(x_test):
    """Linear predictor for XOR data using RBF kernel with sigma = 4."""
    alpha = svm_solver(X, y, 0.1, 10000, hw2_utils.rbf(4))
    return svm_predictor(alpha, X, y, x_test, hw2_utils.rbf(4))