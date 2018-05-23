# -------------------------------------------------------#
# Auxiliar functions for exporting elimination trees
# -------------------------------------------------------#

cimport numpy
import numpy


def greedy_tree_width(adj,method='fill'):
    """ Obtains the ordering that produces the smallest number of fill_in edges in each step
    Args:
        adj (numpy.ndarray): Adjacency matrix
        method (str): 'degree' for min_degree and 'fill' for min_fill
    Return:
        1D-np.array best ordering
        int tree-width for the selected ordering
    """
    cdef numpy.ndarray adj_mor, len_clusters, remaining_vars, best_order
    cdef int i, i_rem, x_rem, tw, j, k

    len_clusters = numpy.zeros(len(adj),dtype=numpy.uint16)
    best_order = numpy.zeros(len(adj),dtype=numpy.uint)
    remaining_vars = numpy.array(range(len(adj)),dtype=numpy.uint)
    adj_mor = moralize(adj)
    for i in range(len(adj)):
        if method == 'fill':
            i_rem = min_fill(adj_mor) #Select node to remove
        elif method == 'degree':
            i_rem = min_degree(adj_mor) #Select node to remove
        else:
            raise ValueError('greedy_tree_width: Undefined method')
        x_rem = remaining_vars[i_rem]
        #Add fill in edges

        parents = numpy.where(adj_mor[:,i_rem])[0]
        len_clusters[x_rem] = len(parents) + 1
        for j in parents:
            for k in parents:
                adj_mor[j,k] = 1
                adj_mor[k,j] = 1
        numpy.fill_diagonal(adj_mor,0)
        
        #remove
        remaining_vars = numpy.delete(remaining_vars,i_rem)
        adj_mor = numpy.delete(adj_mor, i_rem, axis=0)
        adj_mor = numpy.delete(adj_mor, i_rem, axis=1)
        best_order[i] = x_rem

    tw = numpy.max(len_clusters) - 1
    return best_order, tw


def min_degree(adj):
    """ Obtains the ordering that produces the smallest number of fill_in edges in each step
    Args:
        adj (numpy.ndarray): Adjacency matrix
    Return:
        int node to remove
    """
    cdef int i, xi, degree, min_degree

    min_degree = len(adj)
    xi = 0
    for i in range(len(adj)):
        parents = numpy.where(adj[:,i])[0]
        degree = len(parents)
        if degree < min_degree:
            xi = i
            min_degree = degree
    return xi



def min_fill(numpy.ndarray adj):
    """ Obtains the ordering that eliminates the variable with smallest degree in each step
    Args:
        adj (numpy.ndarray): Adjacency matrix
    Return:
        int node to remove
    """
    cdef int i, j, k, xi, num_edges, min_edges

    min_edges = len(adj)
    xi = 0

    for i in range(len(adj)):
        parents = numpy.where(adj[:,i])[0]
        num_edges = 0
        for j in range(len(parents)-1):
            for k in range(j+1,len(parents)):
                if adj[parents[j],parents[k]] == 0:
                    num_edges += 1
        if num_edges < min_edges:
            xi = i
            min_edges = num_edges

    return xi




def moralize(numpy.ndarray adj):
    """ Moralizes a directed graph
    Args:
        adj (numpy.ndarray): Directed adjacency matrix
    Return:
        2D-numpy.ndarray moralized adjacency matrix
    """
    cdef int i,j
    cdef numpy.ndarray adj_mor, parents

    adj_mor = numpy.copy(adj)
    for i in range(len(adj)):
        parents = numpy.where(adj[:,i])[0]
        for j in parents:
            adj_mor[i,j] = 1
            for k in parents:
                adj_mor[j,k] = 1
                adj_mor[k,j] = 1
    numpy.fill_diagonal(adj_mor,0)

    return adj_mor

