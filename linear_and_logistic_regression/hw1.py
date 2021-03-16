import torch
import hw1_utils as utils
import matplotlib.pyplot as plt

'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else, and make sure your feature-expanded matrix in Problem 4 is in the
    specified order (otherwise, your w will be ordered differently than the
    reference solution's in the autograder).
'''

# Problem 3
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    n_by_d = X.shape
    X_new = torch.ones(n_by_d[0], 1)
    X = torch.cat((X_new, X), 1)
    X_transpose = torch.t(X)
    w = torch.zeros(n_by_d[1] + 1, 1)
    ratio = lrate / n_by_d[0]
    
    for i in range(0, num_iter):
        tempvar = torch.matmul(X, w)
        tempvar = torch.sub(tempvar, Y)
        tempvar = torch.matmul(X_transpose, tempvar)
        w = torch.sub(w, tempvar, alpha=ratio)
    return w

def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    X_new = torch.ones(X.shape[0], 1)
    X = torch.cat((X_new, X), 1)
    return torch.matmul(torch.pinverse(X), Y)

def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_reg_data()
    plt.scatter(X, Y, s=None, color='black')
    y = linear_normal(X, Y)
    Z = X
    n_by_d = Z.shape
    Z_new = torch.ones(n_by_d[0], 1)
    Z = torch.cat((Z_new, Z), 1)
    plt.plot(X, torch.matmul(Z, y), 'g')
    myplot = plt.gcf()
    myplot.savefig('3c.png')
    

# Problem 4
def poly_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float): the learning rate
        num_iter (int): number of iterations of gradient descent to perform

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    n_by_d = X.shape
    X = torch.cat((X, X*X), 1)
    for j in range(0, n_by_d[1] - 1):
        for k in range(j+1, n_by_d[1]):
            j_col = torch.FloatTensor([[X[i][j]] for i in range(0, n_by_d[0])])
            k_col = torch.FloatTensor([[X[i][k]] for i in range(0, n_by_d[0])])
            X = torch.cat((X, j_col * k_col), 1)
    X_new = torch.ones(n_by_d[0], 1)
    X = torch.cat((X_new, X), 1)
    X_transpose = torch.t(X)
    w = torch.zeros(1 + n_by_d[1] + (((n_by_d[1]) * (n_by_d[1] + 1)) / 2), 1)
    ratio = lrate / n_by_d[0]
    
    for i in range(0, num_iter):
        tempvar = torch.matmul(X, w)
        tempvar = torch.sub(tempvar, Y)
        tempvar = torch.matmul(X_transpose, tempvar)
        w = torch.sub(w, tempvar, alpha=ratio)
    return w

def poly_normal(X,Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    n_by_d = X.shape
    X_new = torch.ones(n_by_d[0], 1)
    X = torch.cat((X, X*X), 1)
    for j in range(0, n_by_d[1] - 1):
        for k in range(j+1, n_by_d[1]):
            j_col = torch.FloatTensor([[X[i][j]] for i in range(0, n_by_d[0])])
            k_col = torch.FloatTensor([[X[i][k]] for i in range(0, n_by_d[0])])
            X = torch.cat((X, j_col * k_col), 1)
            
    X = torch.cat((X_new, X), 1)
    return torch.matmul(torch.pinverse(X), Y)

def plot_poly():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_reg_data()
    plt.scatter(X, Y, s=None, color='black')
    y = poly_normal(X, Y)
    Z = X
    n_by_d = Z.shape
    Z_new = torch.ones(n_by_d[0], 1)
    Z = torch.cat((Z, Z*Z), 1)
    for j in range(0, n_by_d[1] - 1):
        for k in range(j+1, n_by_d[1]):
            j_col = torch.FloatTensor([[Z[i][j]] for i in range(0, n_by_d[0])])
            k_col = torch.FloatTensor([[Z[i][k]] for i in range(0, n_by_d[0])])
            Z = torch.cat((Z, j_col * k_col), 1) 
    Z = torch.cat((Z_new, Z), 1)
    plt.plot(X, torch.matmul(Z, y), 'g')
    myplot = plt.gcf()
    myplot.savefig('polynomial_regression_fit.png')
    
    plt.show()

A, B = utils.load_xor_data()
B = B.reshape(-1, 1)
def lin_normal_pred(X):
    Z = X
    Z_new = torch.ones(Z.shape[0], 1)
    Z = torch.cat((Z_new, Z), 1)
    return torch.matmul(Z, linear_normal(A, B))

def poly_normal_pred(X):
    Z = X
    n_by_d = Z.shape
    Z_new = torch.ones(n_by_d[0], 1)
    Z = torch.cat((Z, Z*Z), 1)
    for j in range(0, n_by_d[1] - 1):
        for k in range(j+1, n_by_d[1]):
            j_col = torch.FloatTensor([[Z[i][j]] for i in range(0, n_by_d[0])])
            k_col = torch.FloatTensor([[Z[i][k]] for i in range(0, n_by_d[0])])
            Z = torch.cat((Z, j_col * k_col), 1) 
    Z = torch.cat((Z_new, Z), 1)
    return torch.matmul(Z, poly_normal(A, B))

def poly_xor():
    '''
    Returns:
        n x 1 FloatTensor: the linear model's predictions on the XOR dataset
        n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
    ''' 
    Z = A
    n_by_d = Z.shape
    Z_new = torch.ones(n_by_d[0], 1)
    Z = torch.cat((Z_new, Z), 1)
    lin_labels = torch.matmul(Z, linear_normal(A, B))
    
    xmin = -60
    xmax = 60
    ymin = -100
    ymax = 100
    
    K = A
    n_by_d = K.shape
    K_new = torch.ones(n_by_d[0], 1)
    K = torch.cat((K, K*K), 1)
    for j in range(0, n_by_d[1] - 1):
        for k in range(j+1, n_by_d[1]):
            j_col = torch.FloatTensor([[K[i][j]] for i in range(0, n_by_d[0])])
            k_col = torch.FloatTensor([[K[i][k]] for i in range(0, n_by_d[0])])
            K = torch.cat((K, j_col * k_col), 1) 
    K = torch.cat((K_new, K), 1)
    
    poly_labels = torch.matmul(K, poly_normal(A, B))
    
    # comment one of the two contour calls below to show the other contour call's contour plot
    #mycontour = utils.contour_plot(xmin, xmax, ymin, ymax, lin_normal_pred)
    mycontour = utils.contour_plot(xmin, xmax, ymin, ymax, poly_normal_pred)
    mycontour.savefig('4e_lin_normal_pred.png')
    
    plt.show()
    
    return lin_labels, poly_labels
    
# Problem 5
def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    n_by_d = X.shape
    X_new = torch.ones(n_by_d[0], 1)
    X = torch.cat((X_new, X), 1)
    w = torch.zeros(n_by_d[1] + 1, 1)
    ratio = lrate / n_by_d[0]
    
    for k in range(0, num_iter):
        cur = torch.zeros(n_by_d[1] + 1, 1)
        for i in range(0, n_by_d[0]):
            var_i = Y[i]*(torch.matmul(X[i], w)[0])
            var_i = torch.FloatTensor([(var_i).item()])
            var_i = (Y[i] / (1 + ((torch.exp(var_i)[0]).item())))[0].item()
            cur = torch.add(cur, torch.reshape(X[i], (n_by_d[1] + 1, 1)), alpha=var_i)
        w = torch.add(w, cur, alpha=ratio)    
            
    return w

C, D = utils.load_logistic_data()
def lin_gd_second_pred(X):
    Z = X
    Z_new = torch.ones(Z.shape[0], 1)
    Z = torch.cat((Z_new, Z), 1)
    return torch.matmul(Z, linear_gd(C, D))

def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    w_logistic = logistic(C, D)
    w_lin_gd = lin_gd_second_pred(C);
    
    xmin = torch.min(C) - 1
    xmax = torch.max(C) + 1
    ymin = torch.min(D) - 1
    ymax = torch.max(D)
    
    x_logistic = torch.zeros(1, 2)
    x_lin_gd = torch.zeros(1, 2)
    
    if (w_logistic[0][0].item() == 0):
        x_logistic[0][0] = 1
    elif (w_logistic[1][0].item() == 0):
        x_logistic[0][1] = 1
    elif (w_logistic[0][0].item() != 0 and w_logistic[1][0].item() != 0):
        x_logistic[0][0] = -w_logistic[1][0] / w_logistic[0][0]
        x_logistic[0][1] = 1
    
    if (w_lin_gd[0][0].item() == 0):
        x_lin_gd[0][0] = 1
    elif (w_lin_gd[1][0].item() == 0):
        x_lin_gd[0][1] = 1 
    elif (w_lin_gd[0][0].item() != 0 and w_lin_gd[1][0].item() != 0):
        x_lin_gd[0][0] = -w_lin_gd[1][0] / w_lin_gd[0][0]
        x_lin_gd[0][1] = 1
    
    plt.scatter(torch.tensor([C[i][0] for i in range(0, C.shape[0])]).float(), torch.tensor([C[i][1] for i in range(0, C.shape[0])]).float())
    myfig = 0
    if (x_logistic[0][0] == 0):
        plt.plot(torch.linspace(0, 0, 100), torch.linspace(ymin - 1, ymax + 1, 100), 'g')
        myfig = plt.gcf()
    else:
        myx = torch.linspace(xmin - 1, xmax + 1, 100)
        y = (x_logistic[0][1] / x_logistic[0][0]) * myx
        plt.plot(myx, y, 'g')
        myfig = plt.gcf()
    
    if (x_lin_gd[0][0] == 0):
        plt.plot(torch.linspace(0, 0, 100), torch.linspace(ymin - 1, ymax + 1, 100), 'y')
        myfig = plt.gcf()
    else:
        myx_second = torch.linspace(xmin - 1, xmax + 1, 100)
        my_y = (x_lin_gd[0][1] / x_lin_gd[0][0]) * myx_second
        plt.plot(myx_second, my_y, 'y' )
        myfig = plt.gcf()
    
    myfig.savefig('5c.png')
    plt.show()

#W, N = poly_xor()
#plot_linear()
plot_poly()
#logistic_vs_ols()