from libcpp.vector cimport vector
from libcpp.algorithm cimport copy
from libcpp.memory cimport shared_ptr, weak_ptr
from data_type cimport data as datat
from cython.operator cimport dereference as deref, preincrement as inc
cimport numpy as np
import numpy as np
from libcpp cimport bool
cdef extern from "variable_elimination.h":
    cdef cppclass Factor:
        Factor() except +
        Factor(vector[int] variables, vector[int] num_categories) except +
        void learn_params(const int* data, int nrows, int ncols, double alpha)
        double learn_params_se(vector[double] se, double alpha)
        double log_lik_se(vector[double] se, double alpha)
        vector[int] variables 
        vector[int] num_categories
        vector[double] prob # Array with the probabilities 
        vector[vector[int]] mpe # Most-probable instances 
    
    cdef cppclass Factor_Node:
        Factor_Node() except +
        Factor_Node(vector[int] variables, vector[int] num_categories, int id) except +
        Factor_Node(const Factor_Node& fnode) except +
        Factor_Node(shared_ptr[Factor] factor, int id) except +
        Factor_Node(int id) except +
    
        shared_ptr[Factor] factor
        shared_ptr[Factor] joint_factor
        vector[weak_ptr[Factor_Node]] children
        weak_ptr[Factor_Node] parent 
        int id
      
    cdef cppclass Factor_Tree:
        Factor_Tree() except +
        Factor_Tree(const Factor_Tree& ft) except +
        Factor_Tree(vector[int] parents_inner, vector[int] parents_leave, vector[vector[int]] vec_variables_leave, vector[int] num_categories) except +
        void learn_parameters(const int* data, int nrows, int ncols, double alpha)
        void max_compute(int xi)
        void sum_compute(int xi)
        void distribute_evidence()
        void distribute_leaves(vector[int] variables, vector[int] values, int ncols);
        void prod_compute(int xi)
        void max_compute_tree()
        void sum_compute_tree()
        void sum_compute_query(const vector[bool] variables)
        void set_evidence(const vector[int] variables, const vector[int] values)
        void retract_evidence()
        void mpe_data(int* data_complete, const int* data_missing, int ncols, int nrows, vector[int] rows, vector[double] &prob_mpe)
        vector[double] se_data(int* data_missing, int ncols, int nrows, vector[int] rows, vector[int] weights, vector[int] cluster)
        vector[double] se_data_parallel(int* data_missing, int ncols, int nrows, vector[int] rows, vector[int] weights, vector[int] cluster)
        vector[vector[double]] se_data_distribute(int* data_missing, int ncols, int nrows, vector[int] rows, vector[int] weights)
        vector[vector[float]] pred_data(int* data, int ncols, int nrows, vector[int] qvalues, vector[int] query, vector[int] evidence)
        double cll(int* data, int ncols, int nrows, vector[int] query, vector[int] weights)
        double cll_parallel(int* data, int ncols, int nrows, vector[int] query, vector[int] weights)
        double ll_parallel(int* data, int ncols, int nrows)

        
        vector[shared_ptr[Factor_Node]] inner_nodes, leave_nodes
        shared_ptr[Factor_Node] root
    
    shared_ptr[Factor] sum_factor(const Factor &f1, int xi)
    shared_ptr[Factor] prod_factor(const vector[shared_ptr[Factor]] factors)
    shared_ptr[Factor] max_factor(const Factor &f1, int xi)
    double log_lik_se(vector[double] se, double alpha, vector[int] cluster, vector[int] num_categories)

cdef class PyFactor:
    cdef Factor c_factor
    
cdef class PyFactorTree:
    cdef Factor_Tree c_factor_tree 
    