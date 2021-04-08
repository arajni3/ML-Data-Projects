import torch
import hw4_utils


def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [2, N].
        init_c: initial centroids, shape [2, 2]. Each column is a centroid.
    
    Return:
        c: shape [2, 2]. Each column is a centroid.
    """

    if X is None:
        X, init_c = hw4_utils.load_data()
      
    # Note that K = init_c.shape[1] = 2
       
    # Iterate k-Means updates
    for i in range(0, n_iters):
            
        # Save variables mu_1, mu_2 to update the cluster centers later
        # and compare with the previous two cluster centers to check for
        # convergence
        mu_1 = torch.zeros(init_c.shape[0])
        mu_2 = torch.zeros(init_c.shape[0])
        
        # Find responsibilities for each data point and find the set of points in each cluster
        x1 , x2 = [], []
        for j in range(0, X.shape[1]):
            mindist, argmin = -1, -1
            for k in range(0, init_c.shape[1]):
                curdist = torch.dist(init_c[:,k], X[:,j]).item()
                if (mindist<0 or curdist<mindist):
                    mindist = curdist
                    argmin = k
            for k in range(0, init_c.shape[1]):
                if (k == argmin):
                    if (k == 0):
                        x1.append(X[:,j])
                    else:
                        x2.append(X[:,j])
        
        # Get x-coordinates of points in 1st cluster
        x11 = [x1[i][0] for i in range(0, len(x1))]
        # Get y-coordinates of points in 1st cluster
        x12 = [x1[i][1] for i in range(0, len(x1))]
        # Get x-coordinates of points in 2nd cluster
        x21 = [x2[i][0] for i in range(0, len(x2))]
        # Get y-coordinates of points in 2nd cluster
        x22 = [x2[i][1] for i in range(0, len(x2))]
        # Change x1 and x2 into matrices of points that contain exactly the points in the 1st cluster
        # and the points in the 2nd cluster, respectively
        x1 = torch.FloatTensor([x11, x12])
        x2 = torch.FloatTensor([x21, x22])
        
        # Plot clusterings
        c1 = torch.FloatTensor([[init_c[0][0]], [init_c[1][0]]])
        c2 = torch.FloatTensor([[init_c[0][1]], [init_c[1][1]]])
        hw4_utils.vis_cluster(c1, x1, c2, x2, i)
        
        # Find new kth cluster center for all k
        sum_vector = torch.zeros(X.shape[0])
        for j in range(0, x1.shape[1]):
            sum_vector += x1[:,j]
        mu_1 = (1/(x1.shape[1]))*(sum_vector)
        
        sum_vector = torch.zeros(X.shape[0])
        for j in range(0, x2.shape[1]):
            sum_vector += x2[:,j]
        mu_2 = (1/(x2.shape[1]))*(sum_vector)
        
        # Check if algorithm converged, and if it did, then output necessary information and halt the algorithm
        if ((torch.eq(mu_1, init_c[:,0]).all().item()) and (torch.eq(mu_2, init_c[:,1]).all().item())):
            print(f"Number of updates needed for convergence was {i}")
            output = 0
            for j in range(0, x1.shape[1]):
                output += torch.dist(init_c[:,0], x1[:,j]).item()
            for j in range(0, x2.shape[1]):
                output += torch.dist(init_c[:,1], x2[:,j]).item()
                
            print(f"Algorithm converged to cost function value of {(1/2)*(output)}")
            print(f"Cluster centroids ended up being {mu_1} and {mu_2}")
            return init_c
        
        # Else if not converged yet, then update cluster centers
        init_c[:,0] = mu_1
        init_c[:,1] = mu_2        
        
        # Check if algorithm did not converge after n_iters iterations (hence there was too much numerical imprecision)
        if (i == (n_iters) - 1):
            print("Due to numerical imprecision, algorithm did not converge")
            return init_c

k_means()


                
                
