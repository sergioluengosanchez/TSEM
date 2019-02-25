# -------------------------------------------------------#
# Auxiliar functions for exporting elimination trees
# -------------------------------------------------------#
import numpy


def get_adjacency_matrix_from_et(et, debug = False):
    """Gets the adjacency matrix of the Bayesian network encoded by the elimination tree et

    Args:
        et (elimination_tree.ElimTree): Elimination tree

    Returns:
        numpy.array adjacency matrix
    """
    am = numpy.zeros([et.nodes.num_nds, et.nodes.num_nds], dtype=numpy.uint8)
    for i in range(et.nodes.num_nds):
        parents = et.nodes[i].parents.display()
        if debug:
            print 'node ', i, parents
        for j in range(len(parents)):
            am[parents[j], i] = 1
    return am


