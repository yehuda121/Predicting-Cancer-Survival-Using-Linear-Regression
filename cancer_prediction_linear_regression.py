import numpy as np
import matplotlib.pyplot as plt
import time

# Read data from file
def read_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    np.random.shuffle(data)
    X = data[:,0:-1] # Features
    y = data[:, -1]   # Target
    return X, y

def normalize_data(X):
    # Calculate the mean of the dataset along each feature dimension
    mean = np.mean(X, axis=0)
    # Calculate the standard deviation of the dataset along each feature dimension
    std = np.std(X, axis=0)
    # Handle division by zero
    std_nonzero = std.copy()  # Create a copy of the standard deviation array
    std_nonzero[std_nonzero == 0] = 1  # Replace zero std with 1 to avoid division by zero
    # Normalize data
    # If std_nonzero is not zero, proceed with normalization, else just subtract the mean to center the data
    X_normalized = np.where(std_nonzero != 0, (X - mean) / std_nonzero, X - mean)
    
    # Check that after normalization the mean is indeed 0 and the standard deviation is 1
    X_std = np.std(X_normalized, axis=0)
    X_mean = np.mean(X_normalized, axis=0)
    print('check the standard deviation is 1')
    print(X_std)
    print('chek the mean that is 0')
    print(X_mean)
    
    return X_normalized

# Add a column of ones to X matrix
def add_ones_column(X):
    ones_column = np.ones((X.shape[0], 1))
    X_with_ones = np.concatenate((ones_column, X), axis=1)
    return X_with_ones

# Hypothesis function
def hypothesis(theta, X):
    return X @ theta.T

# Cost function
def cost_function(theta, X, y):
    # Determine the number of training examples
    m = 1 if isinstance(y, (int, float, np.float64)) else len(y)
    # Calculate the hypothesis
    h = hypothesis(theta, X)
     # Calculate the cost function J(theta)
    J = (1/(2*m)) * np.sum((h - y)**2)
    # Return the cost
    return J

def gradient_function(theta, X, y):
    # Determine the number of training examples
    m = 1 if isinstance(y, (int, float, np.float64)) else len(y)
    # Calculate the hypothesis
    h = hypothesis(theta, X)
    # Calculate the gradient
    gradient = (1/m) * np.dot(X.T, (h - y))
    # Return the gradient
    return gradient
  
# Gradient descent function
def SGD(X, y, theta, learning_rate, iterations, epsilon=1e-6):
    costs = []  # List to store the cost at each iteration
     # Iterate through a fixed number of iterations
    for _ in range(iterations):
        # Calculate the hypothesis values using current parameters teta and input features X
        prediction = hypothesis(X, theta) #prediction = H(theta)
        
        # Calculate the gradient of the cost function with respect to parameters teta
        gradient = gradient_function(theta, X, y)
        prevTheta = theta
        theta = prevTheta - learning_rate * gradient
        
        # Calculate and store the cost for the current parameters theta
        prev_cost = cost_function(prevTheta, X, y)
        current_cost = cost_function(theta, X, y)
        
        costs.append(current_cost)

        # Check if theta has converged
        if np.linalg.norm(theta - prevTheta) < epsilon:
            break
        # Check if cost has converged
        if np.abs(current_cost - prev_cost) < epsilon:
            break
    return theta, costs

# Mini-batch gradient descent function
def mini_batch_gradient_descent(theta, X, y, alpha, batch_size):
    m = len(y)
    costs = []# List to store the cost at each iteration
    # Iterate over the specified number of iterations
    for i in range(0, m, batch_size):
        X_batch = X[i:i+batch_size]  # Extract mini-batch of features
        y_batch = y[i:i+batch_size]  # Extract mini-batch of target values
        # Calculate the hypothesis for the mini-batch
        h = hypothesis(theta, X_batch)
        # Calculate the gradient for the mini-batch
        gradient = (1/batch_size) * np.dot(X_batch.T, (h - y_batch))
        # Update the parameters using gradient descent
        theta -= alpha * gradient
        # Calculate and store the cost at each iteration
        costs.append(cost_function(theta, X, y))
    # Return the optimized parameters and the list of costs
    return theta, costs

# Singular Value Decomposition
def svd(X, num_features = 3):
    # Perform SVD on X
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    X_reduced = np.dot(U[:, :num_features], np.diag(S[:num_features]))
    return X_reduced


def main():
    # Read data from file
    X, y = read_data("cancer_data.csv")
    
    # Normalize the data
    X_normalized = normalize_data(X)
    
    # Add a column of ones to X matrix
    X_with_ones = add_ones_column(X_normalized)
    
    # Initialize theta
    theta = np.zeros(X_with_ones.shape[1])
    
    # Testing Gradient Descent
    print("\nTesting Gradient Descent:")
    alpha_values = [0.001, 0.01, 0.1, 1]
    for alpha in alpha_values:
        plt.figure()  # Create a new figure
        start_time_sgd = time.time()
        theta_sgd, costs_sgd = SGD(X_with_ones, y, theta, alpha, 30)
        end_time_sgd = time.time()
        print("Runtime for SGD with alpha =", alpha, ":", end_time_sgd - start_time_sgd, "seconds")
        plt.plot(costs_sgd, label=f'alpha={alpha}')
        plt.title("Gradient Descent")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.legend()
        plt.show()
    
    # Testing Mini-Batch Gradient Descent
    print("\nTesting Mini-Batch Gradient Descent:")
    batch_sizes = [1, 10]
    for batch_size in batch_sizes:
        plt.figure()
        start_time_miniBatch = time.time()
        theta_miniBatch, costs_miniBatch = mini_batch_gradient_descent(theta, X_with_ones, y, 0.001, batch_size)
        end_time_miniBatch = time.time()
        print("Runtime for mini-batch gradient descent with batch size =", batch_size, ":", end_time_miniBatch - start_time_miniBatch, "seconds")
        plt.plot(costs_miniBatch, label=f'batch_size={batch_size}')
        plt.title("Mini-Batch Gradient Descent")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.legend()
        plt.show()
    
    '''observation
    The Mini Batch graph descends faster than the Gradient Descent
    '''
    
    # Testing SVD
    print("\nTesting SVD:")
    alphaSvd = [1, 0.1, 0.01, 0.001]
    for alpha in alphaSvd:
        theta_resized = theta[:3]  # Resize theta to size 3
        X_reduced = svd(X_with_ones)
        start_time_sgd = time.time()
        theta_sgd, costs_sgd = SGD(X_reduced, y, theta_resized, alpha, 1000)
        end_time_sgd = time.time()
        plt.figure()  # Create a new figure
        print("Runtime for SGD after reducing using SVD:", end_time_sgd - start_time_sgd, "seconds")
        plt.plot(costs_sgd, label=f'alpha={alpha}')
        plt.title("Gradient Descent after SVD")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.show()
    '''observation
    the run time of the the SGD after reducing using svd is faster than SGD without reducing
    '''

    
if __name__ == "__main__":
    main()
