#cython: boundscheck=False, wraparound=False, nonecheck=False
import elimination_tree
from data_type import ll_metric
import multiprocessing
from data_type cimport data
import numpy
#from elimination_tree cimport ElimTree
cimport numpy as np
from scoring_functions import score_function
from cpython cimport bool
from cython.parallel import prange, parallel, threadid
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from etree cimport PyEliminationTree
import data_type
from copy import deepcopy
from var_elim import PyFactorTree

cdef extern from "log_likelihood.c":
    double logLik(const int* data, int nrows, int ncols, const int* vars, const int* ncat, int l_vars) nogil
cdef extern from "math.h":
    double log(double x) nogil

cdef extern from "elim_tree.h":
      
    cdef cppclass Elimination_Tree nogil:
        #constructors
        Elimination_Tree() except +
        Elimination_Tree(int n) except +
        Elimination_Tree(const Elimination_Tree& et) except +
  
        #Methods
        void set_parent(int xp, int xi, bool inner) nogil
        vector[int] get_pred(int xi, bool inner) nogil
        vector[int] add_arc(int xout,int xin) nogil
        vector[int] remove_arc(int xout,int xin) nogil
        void compute_cluster(int xi) nogil
        void compute_clusters() nogil
        int tw() nogil
        int size() nogil
        bool soundness_node(int xi) nogil
        bool soundness() nogil
        bool completeness_node(int xi, bool inner) nogil
        bool completeness() nogil
        void swap(int xi) nogil
        void optimize(vector[int] xopt) nogil
        bool check_clusters() nogil


        
        vector[vector[int]] cluster_inner, cluster_leaves, children_inner, children_leaves
        vector[int] parents_inner, parents_leaves


cdef np.ndarray get_adjacency_from_et(object et):
    cdef int i,j
    cdef int num_nds = et.nodes.num_nds
    cdef np.ndarray adj = numpy.zeros([num_nds,num_nds],dtype = numpy.int32)
    for i in range(et.nodes.num_nds):
        for j in et.nodes[i].parents.display():
            adj[i,j] = 1
    return adj


# Get matrix with allowed changes. TODO: Check for more efficient ways if necessary
def get_possible_changes(et, list forbidden_parents, int u=10, add_only=False):
    # Possible arc additions
    cdef int num_nds = et.nodes.num_nds
    cdef int i,j
    cdef list pred = [et.get_preds_bn(i) for i in range(num_nds)]
    cdef list desc = [et.get_desc_bn(i) for i in range(num_nds)]
    add_mat = numpy.zeros([num_nds,num_nds],dtype = numpy.int32) + 1
    remove_mat = numpy.zeros([num_nds,num_nds],dtype = numpy.int32)
    reverse_mat = numpy.zeros([num_nds,num_nds],dtype = numpy.int32)
    numpy.fill_diagonal(add_mat,0)

    for i in xrange(num_nds):
        prd = list(pred[i])
        add_mat[i,prd] = 0
        add_mat[et.nodes[i].parents.display(),i] = 0
        add_mat[i,i] = 0
        if et.nodes[i].parents.len_py() >= u:
            add_mat[:,i] = 0
        else:
            add_mat[forbidden_parents[i],i] = 0
    if not add_only:
        # 2. Possible edges to remove
#        remove_mat = get_adjacency_from_et(et)
        for i in xrange(et.nodes.num_nds):
            for j in et.nodes[i].parents.display():
                remove_mat[j,i] = 1
        
            # 3. Possible edges to reverse

        for i in range(et.nodes.num_nds):
            for j in et.nodes[i].parents.display():
                if len(desc[j].intersection(pred[i]))==0:
                    reverse_mat[j,i] = 1
        for i in range(et.nodes.num_nds):
            if et.nodes[i].parents.len_py() >= u:
                reverse_mat[i,:] = 0
            else:
                reverse_mat[i,forbidden_parents[i]] = 0
    return [add_mat, remove_mat, reverse_mat]
    



    
# Return matrix with required and cached results
def get_changes_from_matrix(lop_mats):
    [add_mat, remove_mat, reverse_mat] = lop_mats
    additions = numpy.argwhere(add_mat)
    removals = numpy.argwhere(remove_mat)
    reversals = numpy.argwhere(reverse_mat)
    return [additions, removals, reversals]
    

cdef double score_function_cy( int* data, int nrows, int  nvars, int* variables, int* ncat, int l_vars, int stype) nogil:
    cdef double score
    cdef long b
    cdef int i
    score = logLik(data ,nrows, nvars, variables, ncat, l_vars)
    
    # Obtain parameters
    if stype == 0 or stype == 1:
        b = ncat[variables[0]] - 1
        for i in range(1,l_vars):
            b *= ncat[variables[i]]
        #aic
        if stype == 0:
            score -=  b
        else: 
            score -= (1.0 / 2.0) * log(nrows) * b
    return score
    
#
cdef int find_xi (int xi, int* v, int l_v) nogil:
    cdef int i
    for i in range(l_v):
        if v[i] == xi:
            return i
    return l_v - 1
    
    

def compute_scores(data data, object et, object score_old, object additions_in, object removals_in, object reversals_in , object cache_in, str stype_s): 
    cdef double scorei
    cdef int nvars = data.ncols
    cdef int nrows = data.nrows
    cdef int* dat_c = data.dat_c
    cdef int i, j
    cdef list parents
    cdef int *ncat 
    cdef int* variables
    cdef int* nparents 
    cdef int its_add, its_remove, its_reverse
    cdef int xin, xout, idx_xout
    cdef int stype
    cdef np.ndarray score_difs = numpy.zeros(len(additions_in) + len(removals_in) + len(reversals_in), dtype = numpy.double)
    cdef double [:] score_mv = score_difs
    cdef np.ndarray score_old_aux = numpy.array(score_old, dtype=numpy.double)
    cdef double [:] score_old_mv = score_old_aux
    cdef np.ndarray additions = numpy.array(additions_in, dtype=np.dtype("i"))
    cdef int [:,:] additions_mv
    cdef np.ndarray cache = numpy.array(cache_in, dtype=numpy.double)
    cdef double [:,:] cache_mv = cache
    cdef np.ndarray removals = numpy.array(removals_in, dtype=np.dtype("i"))
    cdef int [:,:] removals_mv
    cdef np.ndarray reversals = numpy.array(reversals_in, dtype=np.dtype("i"))
    cdef int [:,:] reversals_mv
    cdef vector[int] variables_compare_v, variables_compare
    cdef double npmax = 1000000

    
    if stype_s == 'aic':
        stype = 0
    elif stype_s == 'bic':
        stype = 1
    else: #ll
        stype = 2
        
    if len(additions_in) > 0:
        additions_mv = additions
    if len(removals_in) > 0:
        removals_mv = removals
    if len(reversals_in) > 0:
        reversals_mv = reversals
    
    variables_compare_v.resize(nvars)
    ncat = <int *>malloc(nvars * sizeof(int))
    variables= <int *>malloc(nvars * nvars * sizeof(int))
    nparents = <int *>malloc(nvars * sizeof(int))
    for i in range(nvars):
        parents = et.nodes[i].parents.display()
        nparents[i] = len(parents)
        variables[i*nvars] = i
        for j in range(len(parents)):
            variables[i*nvars + j + 1] = parents[j]
    for i in range(nvars):
        ncat[i] = len(data.classes[i])
    
    # Arc additions
    its_add = len(additions)
    for i in prange(0,its_add, nogil=True):
        xout = additions_mv[i,0]
        xin = additions_mv[i,1]
        
        if cache_mv[xout,xin] != npmax:
            score_mv[i] = cache_mv[xout,xin]
        else:
            variables_compare = variables_compare_v
            for j in range(nparents[xin]+1):
                variables_compare[j] = variables[xin*nvars + j]
            variables_compare[nparents[xin] + 1] = xout
            scorei = score_function_cy(dat_c ,nrows, nvars, variables_compare.data(), ncat, nparents[xin] + 2,stype)
            cache_mv[xout,xin] = scorei - score_old_mv[xin]
#            if nparents[xin]==0 and nparents[xout]==0:
#                cache_mv[xin,xout] = cache_mv[xout,xin]
            score_mv[i] = scorei - score_old_mv[xin]
    
    # Arc removals
    its_remove = len(removals)
    for i in prange(0,its_remove, nogil=True):
        xout = removals_mv[i,0]
        xin = removals_mv[i,1]
        if cache_mv[xout,xin] != npmax:
            score_mv[its_add + i] = cache_mv[xout,xin]
        else:
            variables_compare = variables_compare_v
            for j in range(nparents[xin]+1):
                variables_compare[j] = variables[xin*nvars + j]
            idx_xout = find_xi(xout,&variables_compare[1],nparents[xin])
            variables_compare[1 + idx_xout] = variables_compare[nparents[xin]]
            scorei = score_function_cy(dat_c ,nrows, nvars, variables_compare.data(), ncat, nparents[xin],stype)
            cache_mv[xout,xin] = scorei - score_old_mv[xin]
            score_mv[its_add + i] = scorei - score_old_mv[xin]

    # Arc reversals
    its_reverse = len(reversals)
    for i in prange(0,its_reverse, nogil=True):
        # Same as arc addition, but swapping xout and xin
        xout = reversals_mv[i,1]
        xin = reversals_mv[i,0]
        if cache_mv[xout, xin] != npmax:
            score_mv[its_add + its_remove + i] = cache_mv[xout, xin] + cache_mv[xin, xout]
        else:
            variables_compare = variables_compare_v
            for j in range(nparents[xin]+1):
                variables_compare[j] = variables[xin*nvars +j]
            variables_compare[nparents[xin] + 1] = xout
            scorei = score_function_cy(dat_c ,nrows, nvars, variables_compare.data(), ncat, nparents[xin] + 2,stype)
            cache_mv[xout,xin] = scorei - score_old_mv[xin]
            score_mv[its_add + its_remove + i] = (scorei - score_old_mv[xin]) + cache_mv[xin, xout]

    
    result = score_difs
    free(ncat)
    free(nparents)
    free(variables)
    return score_difs, cache

def get_indexes_int(long[:] names, long[:] index, int l_idx):
    cdef long[:] new_names = numpy.zeros(l_idx, dtype=int)
    cdef int i
    for i in range(l_idx):
        new_names[i] = names[index[i]]
    return numpy.array(new_names, dtype=int)

def get_indexes_double(double[:] names, long[:] index, int l_idx):
    cdef double[:] new_names = numpy.zeros(l_idx, dtype=int)
    cdef int i
    for i in range(l_idx):
        new_names[i] = names[index[i]]
    return numpy.array(new_names, dtype=numpy.double)


def get_indexes_int2d(long[:,:] names, long[:] index, int l_idx):
    cdef long[:,:] new_names = numpy.zeros((l_idx,2), dtype=int)
    cdef int i
    for i in range(l_idx):
        new_names[i,:] = names[index[i],:]
    return numpy.array(new_names, dtype=int)    

def sort_and_filter_op(lop, op_names, score_difs, filter_l0 = True):
    if filter_l0:
        flag = [score_difs > 0]
        score_difs_f = score_difs[flag]
        lop_f = lop[flag]
        op_names_f = op_names[flag]
    else: 
        score_difs_f = score_difs
        lop_f = lop
        op_names_f = op_names
    ord_op = numpy.argsort(score_difs_f)[::-1]
    lop1 = lop_f[:,0]
    lop2 = lop_f[:,1]
    lop_f2= numpy.array([lop1[ord_op],lop2[ord_op]]).transpose()
    return lop_f2, op_names_f[ord_op], score_difs_f[ord_op]
    
    
    
def best_pred_cache(data, et, metric, score_old, forbidden_parents, cache, filter_l0 = True, u=10, add_only=False):
    lop_mats = get_possible_changes(et, forbidden_parents, u, add_only)
    lop = get_changes_from_matrix(lop_mats)
    
    score_difs, cache = compute_scores(data,et, score_old, lop[0], lop[1], lop[2] , cache, metric) 
    
    op_add = numpy.full(len(lop[0]),0,dtype=int)
    op_remove = numpy.full(len(lop[1]),1,dtype=int)
    op_reverse = numpy.full(len(lop[2]),2,dtype=int)
    
    op_names = numpy.concatenate([op_add, op_remove, op_reverse])
    lop_o, op_names_o, score_difs_o = sort_and_filter_op(numpy.concatenate(lop), op_names, score_difs, filter_l0 = filter_l0)
    return lop_o, op_names_o, score_difs_o, cache



cdef int psearch_fts(Elimination_Tree etc, int [:] visited_mv, int [:] op_idx_mv, int [:,:] lop_mv, int num_op, int tw_bound, int cores):
    cdef int pos, xout, xin, i
    cdef Elimination_Tree etc2
    cdef vector[int] xopt
    

    for i in prange(num_op, nogil=True,schedule ="static", chunksize=1, num_threads=cores):
        pos = i     

        etc2 = etc
        xout = lop_mv[pos,0]
        xin = lop_mv[pos,1]
        if op_idx_mv[pos] == 0:
            xopt = etc2.add_arc(xout, xin)
            etc2.optimize(xopt)
        elif op_idx_mv[pos] == 1:
            xopt = etc2.remove_arc(xout, xin)
            etc2.optimize(xopt)
        else:
            xopt = etc2.remove_arc(xout, xin)
            etc2.optimize(xopt)
            xopt = etc2.add_arc(xin, xout)
            etc2.optimize(xopt)
        if etc2.tw()<=tw_bound:          
            return pos
        visited_mv[i] = 1
    return num_op

# Search first tractable structure from ordered list of local changes
def search_first_tractable_structure(et, np.ndarray lop, np.ndarray op_names, int tw_bound, list forbidden_parents = None, bool constraint = False, int  cores=multiprocessing.cpu_count()):
    cdef int i,j, first_valid, xout, xin
    cdef Elimination_Tree etc
    cdef PyEliminationTree etc_cy, etc_cy2
    cdef np.ndarray visited = numpy.zeros(len(op_names), dtype=np.dtype("i"))
    cdef int [:] visited_mv = visited
    cdef np.ndarray op_idx = numpy.zeros(len(op_names), dtype=np.dtype("i"))
    cdef int [:] op_idx_mv = op_idx
    cdef np.ndarray lop_n = numpy.array(lop, dtype=np.dtype("i"))
    cdef int [:,:] lop_mv = lop_n
    cdef list removals, reversals, classes, names
    cdef int num_op = len(op_names)
    etc_cy = PyEliminationTree(et.nodes.num_nds)
    etc_cy2 = PyEliminationTree(et.nodes.num_nds)
    etc_cy.comp_from_et(et)
    etc = etc_cy.c_elimination_tree
    
    removals = numpy.where(op_names==1)[0].tolist()
    reversals = numpy.where(op_names==2)[0].tolist()
    for i in removals:
        op_idx_mv[i] = 1
    for i in reversals:
        op_idx_mv[i] = 2
    
    # Search for the first tractable structure
#    print "Entro"
    first_valid = psearch_fts(etc, visited, op_idx, lop_n, num_op, tw_bound, cores)
    # Find first et with bounded tw
    i = 0
    while (i <= first_valid):
        if i == num_op:
            return -1, et
        xout = lop_mv[i,0]
        xin = lop_mv[i,1]
        if visited_mv[i] == 0:
            etc_cy2.c_elimination_tree = etc_cy.c_elimination_tree
            if op_idx_mv[i] == 0:
                etc_cy2.add_arc_opt(xout, xin)
            elif op_idx_mv[i] == 1:
                etc_cy2.remove_arc_opt(xout, xin)
            else:
                etc_cy2.remove_arc_opt(xout, xin)
                etc_cy2.add_arc_opt(xin, xout)
            if etc_cy2.tw()<=tw_bound:
                classes=[]
                names=[]
                for j in range(et.nodes.num_nds):
                    names.append(et.nodes[j].name)
                    classes.append(et.nodes[j].classes)
                et2 = elimination_tree.ElimTree('et', names, classes)
                etc_cy2.comp_to_et(et2)
                return i, et2
        if constraint:
            if op_idx_mv[i] == 0:
                forbidden_parents[xin].append(xout)
            elif op_idx_mv[i] == 2:
                forbidden_parents[xout].append(xin) 
        i += 1
    return -1, et

