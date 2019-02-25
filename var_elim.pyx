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
        Factor(vector[int] &variables, vector[int] &num_categories) except +
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


cdef class PyFactorTree:
#    cdef Factor_Tree c_factor_tree      
    
    def __cinit__(self,list parents_inner, list parents_leave, list vec_variables_leave, list num_categories):
        cdef vector[int] parents_inner_c
        cdef vector[int] parents_leave_c
        cdef vector[vector[int]] vec_variables_leave_c
        cdef vector[int] num_categories_c
        cdef int p, i, v, c
        cdef list v_l
        
        for p in parents_inner:
            parents_inner_c.push_back(p)
        for p in parents_leave:
            parents_leave_c.push_back(p)
        vec_variables_leave_c.resize(len(vec_variables_leave))
        for i, v_l in enumerate(vec_variables_leave):
            for v in v_l:
                vec_variables_leave_c[i].push_back(v)
        for c in num_categories:
            num_categories_c.push_back(c)
        self.c_factor_tree = Factor_Tree(parents_inner_c, parents_leave, vec_variables_leave_c, num_categories_c)
    
    def num_nodes(self):
        return self.c_factor_tree.inner_nodes.size()
    
    def copy(self):
        cdef PyFactorTree fiti_new = PyFactorTree([-1],[0],[[0]],[2])
        fiti_new.c_factor_tree = Factor_Tree(self.c_factor_tree)
        return fiti_new
#    def __cinit__(self):
#        return

    
    def get_parent(self, int id):
        if id == -1:
            return -1
        elif id < self.c_factor_tree.inner_nodes.size() and id >= 0:
            return deref(deref(self.c_factor_tree.inner_nodes[id]).parent.lock()).id
        elif id < 2*self.c_factor_tree.inner_nodes.size() and id >= 0:
            return deref(deref(self.c_factor_tree.leave_nodes[id-self.c_factor_tree.inner_nodes.size()]).parent.lock()).id
        else:
            raise IndexError('id out of bounds')
        
    def get_children(self, int id):
        if id == -1:
            node= deref(self.c_factor_tree.root)
        elif id < self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.inner_nodes[id])
        elif id < 2*self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.leave_nodes[id-self.c_factor_tree.inner_nodes.size()])
        else:
            raise IndexError('id out of bounds')
        ch = []
        for i in range(node.children.size()):
            ch.append(deref(node.children[i].lock()).id)
        return ch
    
    # Set parameters by hand. Params is a list of lists with the parameters
    # of the nodes
    def set_parameters(self, list nodes, list params):
        cdef vector[double] param_vector 
        for node_i, pari in zip(nodes,params):
            param_vector.clear()
            for parij in pari:
                param_vector.push_back(parij)
            deref(deref(self.c_factor_tree.leave_nodes[node_i]).factor).prob = param_vector

    # get parameters 
    def get_parameters(self, list nodes_in = None):
        cdef PyFactor factor
        cdef list probs = []
        cdef int num_nds = self.c_factor_tree.inner_nodes.size()
        if nodes_in is None:
            nodes = range(num_nds)
        else:
            nodes = nodes_in
        for node_i in nodes:
            factor = self.get_factor(node_i+num_nds)
            probs.append(factor.get_prob())
        return probs
            
            
    
    def learn_parameters(self, datat data, double alpha = 0.0):
        self.c_factor_tree.learn_parameters(data.dat_c, data.nrows, data.ncols, alpha)   
    
    
    # Debugging function
    def get_factor(self, int id): 
        if id == -1:
            node= deref(self.c_factor_tree.root)
        elif id < self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.inner_nodes[id])
        elif id < 2*self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.leave_nodes[id-self.c_factor_tree.inner_nodes.size()])
        else:
            raise IndexError('id out of bounds')
        factor = PyFactor([],[])
        factor.c_factor = deref(node.factor)
        return factor
    
    # Debugging function
    def get_joint_factor(self, int id): 
        if id == -1:
            node= deref(self.c_factor_tree.root)
        elif id < self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.inner_nodes[id])
        elif id < 2*self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.leave_nodes[id-self.c_factor_tree.inner_nodes.size()])
        else:
            raise IndexError('id out of bounds')
        factor = PyFactor([],[])
        factor.c_factor = deref(node.joint_factor)
        return factor  
    
    # Debugging function
    def sum_factor(self, int id, int xi): 
        if id < self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.inner_nodes[id])
        elif id < 2*self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.leave_nodes[id-self.c_factor_tree.inner_nodes.size()])
        else:
            raise IndexError('id out of bounds')
        a = sum_factor(deref(node.factor),xi)
        factor = PyFactor([],[])
        factor.c_factor = deref(a)
        return factor
        
    def prod_factor(self, list ids):
        cdef vector[shared_ptr[Factor]] factors
        for id in ids:        
            if id == -1:
                node= deref(self.c_factor_tree.root)
            elif id < self.c_factor_tree.inner_nodes.size() and id >= 0:
                node = deref(self.c_factor_tree.inner_nodes[id])
            elif id < 2*self.c_factor_tree.inner_nodes.size() and id >= 0:
                node = deref(self.c_factor_tree.leave_nodes[id-self.c_factor_tree.inner_nodes.size()])
            else:
                raise IndexError('id out of bounds')
            factors.push_back(node.factor)
        a = prod_factor(factors)
        
        factor = PyFactor([],[])
        factor.c_factor = deref(a)
        return factor
        
    def max_factor(self, int id, int xi): 
        if id < self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.inner_nodes[id])
        elif id < 2*self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.leave_nodes[id-self.c_factor_tree.inner_nodes.size()])
        else:
            raise IndexError('id out of bounds')
        a = max_factor(deref(node.factor),xi)
        factor = PyFactor([],[])
        factor.c_factor = deref(a)
        return factor
        
        
    def max_compute(self, int xi):
        self.c_factor_tree.max_compute(xi)
        
    def sum_compute(self, int xi):
        self.c_factor_tree.sum_compute(xi)
    
    def distribute_evidence(self):
        self.c_factor_tree.distribute_evidence()
    
    def distribute_leaves(self, list var, list val):
        cdef vector[int] variables
        cdef vector[int] values
        cdef int ncols = self.c_factor_tree.inner_nodes.size()
        for p in var:
            variables.push_back(p)
        for p in val:
            values.push_back(p)
        self.c_factor_tree.distribute_leaves(variables, values, ncols)
        
    def prod_compute(self, int xi):
        self.c_factor_tree.prod_compute(xi)
        
    def max_compute_tree(self):
        self.c_factor_tree.max_compute_tree()
        
    def sum_compute_tree(self):
        self.c_factor_tree.sum_compute_tree()
    
    def sum_compute_query(self, list var):
        cdef vector[bool] variables
        for p in var:
            variables.push_back(p)
        self.c_factor_tree.sum_compute_query(variables)
    
    def set_evidence(self, list var, list val):
        cdef vector[int] variables
        cdef vector[int] values
        for p in var:
            variables.push_back(p)
        for p in val:
            values.push_back(p)
        self.c_factor_tree.set_evidence(variables, values)

    def retract_evidence(self):
        self.c_factor_tree.retract_evidence()
        
    def mpe_data(self, datat data_complete, datat data_missing, list rows, return_prob = False):
        cdef vector[int] rows_v
        cdef vector[double] mpe_prob
        cdef vector[double].iterator it_prob
        cdef list l_prob = []

        for r in rows:
            rows_v.push_back(r)
        self.c_factor_tree.mpe_data(data_complete.dat_c, data_missing.dat_c, data_complete.ncols, data_complete.nrows, rows_v, mpe_prob)
        if return_prob:
             it_prob = mpe_prob.begin()
             while it_prob != mpe_prob.end():
                 l_prob.append(deref(it_prob))
                 inc(it_prob)
        return l_prob
        
    def se_data(self, datat data_missing, list rows, list weights, list cluster):
        cdef vector[int] rows_v, weights_v
        cdef vector[int] cluster_v
        cdef vector[double] se
        cdef vector[double].iterator it_prob
        cdef list l_prob = []

        for r in rows:
            rows_v.push_back(r)
        for w in weights:
            weights_v.push_back(w)
        for c in cluster:
            cluster_v.push_back(c)
        se = self.c_factor_tree.se_data(data_missing.dat_c, data_missing.ncols, data_missing.nrows, rows_v, weights_v, cluster_v)
        it_prob = se.begin()
        while it_prob != se.end():
            l_prob.append(deref(it_prob))
            inc(it_prob)
        return l_prob
    

    def EM_parameters_distribute(self, datat data_missing, PyFactorTree fiti_old, list rows = None, list weights=None, double alpha = 0.0):
        cdef vector[int] rows_v, weights_v
        cdef vector[int] cluster_v
        cdef vector[double] se
        cdef vector[vector[double]] v_se
        cdef list ll
        ll = []
        if rows is None:
             for i in range(data_missing.nrows):
                rows_v.push_back(i)       
        else:
            for r in rows:
                rows_v.push_back(r)
        if weights is None:
            for _ in range(data_missing.nrows):
                weights_v.push_back(1)
        else:
            for w in weights:
                weights_v.push_back(w)
        
        v_se= fiti_old.c_factor_tree.se_data_distribute(data_missing.dat_c, data_missing.ncols, data_missing.nrows, rows_v, weights_v)
        for xi in range(data_missing.ncols):
            ll.append(deref(deref(self.c_factor_tree.leave_nodes[xi]).factor).learn_params_se(v_se.at(xi), alpha))
        return ll
    
    def EM_parameters(self, datat data_missing, PyFactorTree fiti_old, list rows = None, list weights = None, double alpha = 0.0, list var_update = None, parallel = False):
        cdef vector[int] rows_v, weights_v
        cdef vector[int] cluster_v
        cdef vector[double] se
        cdef list list_se
        cdef list cluster
        cdef list ll
        cdef list vup
        ll = []
        list_se = []
        if rows is None:
             for i in range(data_missing.nrows):
                rows_v.push_back(i)       
        else:
            for r in rows:
                rows_v.push_back(r)
        if weights is None:
            for _ in range(data_missing.nrows):
                weights_v.push_back(1)
        else:
            for w in weights:
                weights_v.push_back(w)
        
        if var_update is None:
            vup = range(data_missing.ncols)
        else:
            vup = var_update
        
        for xi in vup:
            cluster = self.get_factor(data_missing.ncols+xi).get_variables()
            cluster_v.clear()
            for c in cluster:
                cluster_v.push_back(c)
            if parallel:
                print 'EMp size: ', rows_v.size()
                se = fiti_old.c_factor_tree.se_data_parallel(data_missing.dat_c, data_missing.ncols, data_missing.nrows, rows_v, weights_v, cluster_v)
            else:
                se = fiti_old.c_factor_tree.se_data(data_missing.dat_c, data_missing.ncols, data_missing.nrows, rows_v, weights_v, cluster_v)
            list_se.append(se)
        for i,xi in enumerate(vup):
            ll.append(deref(deref(self.c_factor_tree.leave_nodes[xi]).factor).learn_params_se(list_se[i], alpha))
        return ll
    
    
    
    def learn_parameters_se(self, datat data_missing, list rows, list weights, int xi, double alpha = 0.0):
        cdef vector[int] rows_v, weights_v
        cdef vector[int] cluster_v
        cdef vector[double] se
        cdef list cluster
        cdef double ll

        for r in rows:
            rows_v.push_back(r)
        for w in weights:
            weights_v.push_back(w)
        
        cluster = self.get_factor(data_missing.ncols+xi).get_variables()
        for c in cluster:
            cluster_v.push_back(c)
        
        se = self.c_factor_tree.se_data(data_missing.dat_c, data_missing.ncols, data_missing.nrows, rows_v, weights_v, cluster_v)
        
        ll = deref(deref(self.c_factor_tree.leave_nodes[xi]).factor).learn_params_se(se, alpha)
        return ll
        
             
    def log_lik_se(self, datat data_missing, list rows, list weights, int xi, double alpha = 0.0):
        cdef vector[int] rows_v, weights_v
        cdef vector[int] cluster_v
        cdef vector[double] se
        cdef list cluster
        cdef double ll
        
        for r in rows:
            rows_v.push_back(r)
        for w in weights:
            weights_v.push_back(w)
        
        cluster = self.get_factor(data_missing.ncols+xi).get_variables()
        for c in cluster:
            cluster_v.push_back(c)
        
        se = self.c_factor_tree.se_data(data_missing.dat_c, data_missing.ncols, data_missing.nrows, rows_v, weights_v, cluster_v)
        
        ll = deref(deref(self.c_factor_tree.leave_nodes[xi]).factor).log_lik_se(se, alpha)
        return ll   
    
    def log_lik_se_cluster(self, datat data_missing, list rows, list weights, list cluster, double alpha = 0.0):
        cdef vector[int] rows_v, weights_v
        cdef vector[int] cluster_v
        cdef vector[int] num_categories_v
        cdef vector[double] se
        cdef double ll
        
        num_categories_v = deref(deref(self.c_factor_tree.leave_nodes[0]).factor).num_categories
        for r in rows:
            rows_v.push_back(r)
        for w in weights:
            weights_v.push_back(w)
    
        for c in cluster:
            cluster_v.push_back(c)
        
        se = self.c_factor_tree.se_data(data_missing.dat_c, data_missing.ncols, data_missing.nrows, rows_v, weights_v, cluster_v)
        ll = log_lik_se(se, alpha, cluster_v, num_categories_v)
        return ll   
    
    def pred_data(self, datat data, list qvalues_l, list query_l, list evidence_l):
        cdef vector[int] qvalues, query, evidence
        cdef vector[ vector[float]] response
        cdef int x, i, j
        cdef np.ndarray resp
        resp = np.zeros((data.nrows, len(query_l)))
        
        for x in qvalues_l:
            qvalues.push_back(x)
        for x in query_l:
            query.push_back(x)
        for x in evidence_l:
            evidence.push_back(x)
        
        response = self.c_factor_tree.pred_data(data.dat_c, data.ncols, data.nrows, qvalues, query, evidence)
        
        for i in range(data.nrows):
            for j in range(len(query_l)): 
                resp[i,j] = response[j][i]
        return resp        
    
    
    def cll(self, datat data, list query_l, list weights = None, parallel = False):
        cdef vector[int] query, weights_v
        if weights is None:
            for _ in range(data.nrows):
                weights_v.push_back(1)
        else:
            for w in weights:
                weights_v.push_back(w)
        for x in query_l:
            query.push_back(x)
        if parallel:
            return self.c_factor_tree.cll_parallel(data.dat_c, data.ncols, data.nrows, query, weights_v)
        else:
            return self.c_factor_tree.cll(data.dat_c, data.ncols, data.nrows, query, weights_v)

    def ll(self, datat data):
#        print 'll size: ', data.nrows
        return self.c_factor_tree.ll_parallel(data.dat_c, data.ncols, data.nrows)     
        

cdef class PyFactor:
#    cdef Factor c_factor      # hold a C++ instance which we're wrapping
    
    def __cinit__(self,list variables, list num_categories):
        cdef vector[int] variables_c
        cdef vector[int] num_categories_c
        for v in variables:
            variables_c.push_back(v)
        for c in num_categories:
            num_categories_c.push_back(c)
        self.c_factor = Factor(variables_c, num_categories_c)
    
    def learn_params(self, datat data, double alpha = 0.0):
        self.c_factor.learn_params(data.dat_c, data.nrows, data.ncols, alpha)
        
    # Get variables in the domain of the factor
    def get_variables(self):
        cdef vector[int].iterator it_variables = self.c_factor.variables.begin()
        cdef list l_variables = []
        while it_variables != self.c_factor.variables.end():
            l_variables.append(deref(it_variables))
            inc(it_variables)
        return l_variables
        
    # Get list of probabilities of the factor
    def get_prob(self):
        cdef vector[double].iterator it_prob = self.c_factor.prob.begin()
        cdef list l_prob = []
        while it_prob != self.c_factor.prob.end():
            l_prob.append(deref(it_prob))
            inc(it_prob)
        return l_prob
            
    # Get current mpe from the factor
    def get_mpe(self):
        cdef vector[vector[int]].iterator it_mpe = self.c_factor.mpe.begin()
        cdef vector[int].iterator it_mpe_i
        cdef list l_mpe = []
        cdef list l_mpe_i
        while it_mpe != self.c_factor.mpe.end():
            it_mpe_i = deref(it_mpe).begin()
            l_mpe_i = []
            while it_mpe_i != deref(it_mpe).end():
                l_mpe_i.append(deref(it_mpe_i))
                inc(it_mpe_i)
            l_mpe.append(l_mpe_i)
            inc(it_mpe)
        return l_mpe
    
    # Get the number of categories of the variables in the factor
    def get_num_categories(self):
        cdef vector[int].iterator it_variables = self.c_factor.variables.begin()
        cdef list l_num_categories = []
        while it_variables != self.c_factor.variables.end():
            l_num_categories.append(self.c_factor.num_categories[deref(it_variables)])
            inc(it_variables)
        return l_num_categories
        

# Test efficiency of vector operations    
#def test_vectors(nmul, nsum, len_v, nvec =2):
#    import random
#    from time import time
#    cdef vector[vector[double]] vectors 
#    cdef vector[double] vec
#    cdef vector[double] vec1
#    cdef int i, j
#    cdef double* ar1, *ar2, *ar3
##    cdef np.ndarray narray = numpy.array(len_v, )
#    
#    
#    #Initialize randomly
#    for i in range(nvec):
#        vec.clear()
#        for j in range(len_v):
#            vec.push_back(random.uniform(0, 1))
#        vectors.push_back(vec)
#    
#    ta = time()
#    # test
#    vec.resize(len_v) 
#    for i in range(nmul):
##        vec.resize(0)
#        vec1.reserve(len_v)
##        copy(vec.begin(), vec.end(), vec1.begin())
##        vec1 = vec   
##        ar1 = vec1.data()
##        ar2 = vectors[0].data()
##        ar3 = vectors[1].data()
##        mul_arrays(ar1, ar2, ar3, len_v)
##        vec = mul_vectors(vectors)
#    tb = time()
#    for i in range(nsum):
#        vect = sum_vectors(vectors)
#    tc = time()
#    
#    print 'time mul: ', tb - ta, ', time sum: ', tc - tb
    

        
        
