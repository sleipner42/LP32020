
# %% asd
import pandas as pd
import numpy as np
import math
from scipy.stats import describe, randint
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import scipy.sparse
from scipy.optimize import least_squares

sns.set()

# Methods for exporting tables and images to latex
def saveimg(filename, fig):
    file_path = os.path.join(os.getcwd(), 'images')
    os.makedirs(file_path, exist_ok=True)
    fig.savefig(os.path.join(file_path, filename))
    
def savetable(filename, df):
    file_path = os.path.join(os.getcwd(), 'tables')
    os.makedirs(file_path, exist_ok=True)
    pd.DataFrame(df).to_latex(os.path.join(file_path, filename),escape = False, index = False)

def random_neighbour(points):

    # Getting dimensions of points
    dimension = np.shape(points)
    
    # Generate if going up or down, do this for all points
    up_or_down = 2*np.random.randint(2, size=dimension[0]) - 1
    
    # Generate which diretion to go in for all points
    direction = np.random.randint(dimension[1], size=dimension[0])
    
    # Create new matrix, create a binary matrix of the direction
    indptr = range(len(direction)+1)
    data = np.ones(len(direction))
    points_to_change = scipy.sparse.csr_matrix((data, direction, indptr)).toarray()
    
    # Add the binary matrix multiplied by direction to the old points
    new_points = points + (points_to_change.T*up_or_down).T
    
    # Generating probabilities
    probabilities = np.ones(dimension[0])*1/(dimension[1]*2)
    
    return new_points, probabilities

# %% Methods used in the lab

def random_neighbour_avoiding(random_walks):
    
    # Start by swapping axis i and k so it is possible to iterate over all random walks
    random_walks_list = np.swapaxes(random_walks, 0, 1)
    
    new_points = []
    probabilities = []
    
    # Iterate over random walks
    for random_walk in random_walks_list:
        
        random_walk = random_walk.tolist()
        
        last_point = random_walk[-1]
        viable_points = []
        
        dimension = np.shape(last_point)[0]
        
        for n in range(dimension):
            
            change = np.zeros(dimension)
            change[n] = 1
            
            point_up = last_point - change
            point_down = last_point + change
            
            viable_points.append(list(point_up))
            viable_points.append(list(point_down))
        
        viable_points = list(filter(lambda x: x not in random_walk, viable_points))
        length = len(viable_points)
                        
        if length == 0:
            new_points.append(last_point)
            probabilities.append(0)
        else:   
            probabilities.append(1/length)
            new_point_index = np.random.randint(length, size=1)[0]
            new_points.append(viable_points[new_point_index])
        
    return new_points, probabilities

def is_self_avoiding(random_walk):
    return int(len(np.unique(random_walk, axis=0)) == len(random_walk))

def generate_weights(random_walks, weights, probabilities):
    
    # Get indexes where weights is different from zero
    indexes = np.intersect1d(np.nonzero(weights)[0], np.nonzero(probabilities)[0])
    
    # Create new weights
    new_weights = np.zeros(len(weights))
    
    # Start by swapping axis i and k so it is possible to iterate over all random walks
    random_walks_list = np.swapaxes(random_walks, 0, 1)

    # New weights
    np.take(random_walks_list,indexes)
    #updated_weights = np.apply_along_axis(np.take(random_walks_list,indexes), 0, is_self_avoiding)
    #updated_weights = new_weights/probabilities[indexes]*weights[indexes]
        

        
    return new_weights      

def resample(random_walks, weights):
    
    # Get indexes where weights is different from zero
    zero_index = np.nonzero(weights == 0)[0]
    non_zero_index = np.nonzero(weights)[0]
    
    probabilities = weights/np.sum(weights)
    random_walks_list = np.swapaxes(random_walks, 0, 1)
    
    replacement_index = np.random.choice(np.arange(len(random_walks_list)), len(zero_index), p = probabilities)
    
    for i, index in enumerate(zero_index):
        random_walks_list[index] = random_walks_list[replacement_index[i]]
        weights[index] = 1
    
    return np.swapaxes(random_walks_list, 0, 1).tolist(), weights

def init_walk (d, k, N, self_avoiding = False, do_resample = False):
    
    # The random_walk array has the following structure compared 
    # to the lecture notes random_walk[k][i][d], where:
    # k = step number k
    # i = simulation number
    # d = dimension 
    
    random_walks = []
    weights = [[1] * N]
    
    random_walks.append(np.zeros((N, d)))
    
    for step in range(k):
        
        if self_avoiding:
            step_all, probabilities = random_neighbour_avoiding(random_walks)
        else:
            step_all, probabilities = random_neighbour(random_walks[step])
            
        random_walks.append(step_all)
        weights.append(generate_weights(random_walks, weights[step], probabilities))
        
        if do_resample:
            random_walks, weights[-1] = resample(random_walks, weights[-1])
            
        print("-", end="")
        
    return random_walks, weights

walks, weights = init_walk(10, 20, 10**2, self_avoiding=True, do_resample=True)

print("")

cn = np.mean(weights, axis=1)
length = len(cn)

y = np.log(cn).T
X = np.hstack((np.ones(length), np.arange(length)+1, np.log(np.arange(length)+1))).reshape(3,length).T
theta = np.linalg.inv(X.T@X)@X.T@y
print(np.e**theta)


# %%
