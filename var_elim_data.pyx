from libcpp.vector cimport vector
from libcpp.algorithm cimport copy
from libcpp.memory cimport shared_ptr, weak_ptr, make_shared
from data_type cimport data as datat
from cython.operator cimport dereference as deref, preincrement as inc
cimport numpy as np
import numpy as np
from libcpp cimport bool
from cython.parallel import prange
from sklearn.model_selection import KFold
#import var_elim

cdef extern from "variable_elimination_data.h":
    cdef cppclass Factor_data:
        Factor_data() except +
        Factor_data(vector[int] variables, vector[int] num_categories, unsigned data_size) except +
        Factor_data(const Factor_data &fd) except +
        void set_params_evidence(vector[double] parameters, vector[shared_ptr[vector[vector[int]]]] &indicators, vector[int] num_categories)  nogil
        vector[int] variables 
        double* prob # Array with the probabilities
        unsigned num_conf;
        unsigned data_size;
    vector[double] mul_vectors(vector[vector[double]] vectors_in)
    vector[double] sum_vectors(vector[vector[double]] vectors_in)
    void mul_arrays(double* array_in, double* array1, double* array2, unsigned len)
    vector[shared_ptr[vector[vector[int]]]] generate_indicators(int* data, int nrows, int ncols, vector[int] num_categories)
    vector[shared_ptr[vector[vector[int]]]] generate_indicators(int* data, int nrows, int ncols, vector[int] num_categories, vector[int] variables)
    void example_dlib()
    
    cdef cppclass Factor_Node_data:
        Factor_Node_data() except +
        Factor_Node_data(vector[int] &variables, vector[int] &num_categories, int id) except +
        Factor_Node_data(const Factor_Node_data& fnode) except +
        Factor_Node_data(shared_ptr[Factor_data] factor, int id) except +
        Factor_Node_data(int id) except +
    
        shared_ptr[Factor_data] factor
        shared_ptr[Factor_data] joint_factor
        vector[weak_ptr[Factor_Node_data]] children
        weak_ptr[Factor_Node_data] parent 
        int id

    cdef cppclass Factor_Tree_data:
        Factor_Tree_data() except +
#        Factor_Tree_data(const Factor_Tree_data& ft) except + #Very weird import error
        Factor_Tree_data(vector[int] parents_inner, vector[int] parents_leave, vector[vector[int]] vec_variables_leave, vector[int] num_categories) except +
        void sum_compute_tree(vector[vector[double]] &parameters, vector[shared_ptr[vector[vector[int]]]] &indicators);
        void sum_compute_query(vector[bool] &variables, vector[vector[double]] &parameters, vector[shared_ptr[vector[vector[int]]]] &indicators)
        vector[double] se_data(vector[int] &cluster_int, vector[vector[double]] &parameters, vector[shared_ptr[vector[vector[int]]]] &indicators)
        vector[double] se_data_parallel(vector[int] &cluster_int, vector[vector[double]] &parameters, vector[vector[shared_ptr[vector[vector[int]]]]] &indicators_vector)

        vector[double] em(vector[shared_ptr[vector[vector[int]]]] &indicators, int num_it, double alpha)
        vector[double] em_parallel(vector[vector[shared_ptr[vector[vector[int]]]]] &indicators_vector, int num_it, double alpha)
        double learn_params_se(vector[double] &se, double alpha, unsigned xi)
        void learn_params(const int* data, int nrows, int ncols, double alpha, unsigned xi)
        void learn_params(const int* data, int nrows, int ncols, double alpha)
        double log_likelihood(vector[shared_ptr[vector[vector[int]]]] &indicators)
        double log_likelihood_parallel(vector[vector[shared_ptr[vector[vector[int]]]]] &indicators_vector)
        
        vector[shared_ptr[Factor_Node_data]] inner_nodes, leave_nodes
        shared_ptr[Factor_Node_data] root
        vector[int] num_categories;
        unsigned data_size;
        vector[vector[double]] parameters
        vector[vector[int]] variables 

cdef class Indicators:
    cdef vector[shared_ptr[vector[vector[int]]]] cvalues      
    
    def __cinit__(self, datat data, variables = None):
        cdef vector[int] num_categories, variables_v
        for cl in data.classes:
             num_categories.push_back(len(cl))
        if variables is not None:
            for xi in variables:
                variables_v.push_back(xi)
            self.cvalues = generate_indicators(data.dat_c, data.nrows, data.ncols, num_categories, variables_v)
        else:
            self.cvalues = generate_indicators(data.dat_c, data.nrows, data.ncols, num_categories)        
        
    def print_indicators(self):
        for col in range(self.cvalues.size()):
            col_values = deref(self.cvalues[col])
            for cat in range(col_values.size()):
                cat_values = col_values[cat]
                print 'indicators variable ', col, ', category: ', cat
                val_str = "["
                for row in range(cat_values.size()):
                    val_str = val_str + str(cat_values[row])
                    if row < (cat_values.size()-1):
                        val_str = val_str  + ", "
                val_str = val_str  + "]"
                print val_str
            print
    
    # Selects the subset of indicators that corresponds to variables
    def subset_indicators(self, list variables):
        cdef Indicators ind2 = Indicators()
        for i in variables:
            ind2.cvalues.push_back(self.cvalues[i])
        return ind2
        
        

cdef class Indicators_vector:
    cdef vector[vector[shared_ptr[vector[vector[int]]]]] ind_vector      
    
    def __cinit__(self, object df, list classes, int num_folds, variables = None):
        cdef Indicators ind
        kf = KFold(n_splits=num_folds)
        for _, fold_idx in kf.split(df):
            df2 = df.iloc[fold_idx,:]
            dt2 = datat(df2, classes = classes)
            ind = Indicators(dt2, variables)
            self.ind_vector.push_back(ind.cvalues)
    
     # Selects the subset of indicators that corresponds to variables
    def subset_indicators(self, list variables):
        cdef Indicators ind2 = Indicators()
        cdef Indicators_vector indv2 = Indicators_vector()
        for j in self.ind_vector.size():
            ind2.cvalues.clear()
            for i in variables:
                ind2.cvalues.push_back(self.ind_vector[j][i])
            indv2.ind_vector.push_back(ind2.cvalues)
        return indv2       
        
        

cdef class PyFactor_data:
    cdef Factor_data c_factor_data
    
    def __cinit__(self,list variables, list num_categories, int ninst):
        cdef vector[int] variables_c
        cdef vector[int] num_categories_c
        for v in variables:
            variables_c.push_back(v)
        for c in num_categories:
            num_categories_c.push_back(c)
        if ninst == 0:
            self.c_factor_data = Factor_data()
        else:
            self.c_factor_data = Factor_data(variables_c, num_categories_c, ninst)
        
    def set_params_evidence(self,list parameters, Indicators ind, list num_categories):
        cdef vector[double] parameters_c
        cdef vector[int] num_categories_c
        cdef int c
        for p in parameters:
            parameters_c.push_back(p)
        for c in num_categories:
            num_categories_c.push_back(c)
        self.c_factor_data.set_params_evidence(parameters_c, ind.cvalues, num_categories_c)
        
    def test_params_evidence(self,list parameters, Indicators ind, int reps, list num_categories):
        from time import time
        cdef vector[double] parameters_c
        cdef Factor_data fd = self.c_factor_data
        cdef vector[shared_ptr[vector[vector[int]]]] cvalues = ind.cvalues
        cdef vector[int] num_categories_c
        cdef int i, c
        for c in num_categories:
            num_categories_c.push_back(c)        
        for p in parameters:
            parameters_c.push_back(p)
        ta = time()
#        for i in prange(0,reps, nogil=True, num_threads=8):
        for i in range(reps):
            fd.set_params_evidence(parameters_c, cvalues, num_categories_c)
        tb = time()
        print "time: ", tb - ta
    
    def print_factor(self, data_size, nconf):
        cdef double* prob = self.c_factor_data.prob
        for row in range(data_size):
            print 'Probs for instance ', row
            val_str = "["
            for col in range(nconf):
                val_str = val_str + str(prob[col* data_size+row])
                if col < (nconf-1):
                    val_str = val_str  + ", "
            val_str = val_str  + "]"           
            print val_str
        print
    
        

cdef class PyFactorTree_data:
    cdef Factor_Tree_data c_factor_tree_data      
    
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

        self.c_factor_tree_data = Factor_Tree_data(parents_inner_c, parents_leave, vec_variables_leave_c, num_categories_c)
        
    def sum_compute_tree(self, object pyft, Indicators ind):
        from time import time
        cdef vector[vector[double]] parameters
        cdef vector[double] p_v
        cdef int i,n
        cdef object pyf
        cdef list param
        ta = time()
        n = pyft.num_nodes()
        for i in range(n):
            pyf = pyft.get_factor(n + i)
            param = pyf.get_prob()
            p_v.clear()
            for pi in param:
                p_v.push_back(pi)
            parameters.push_back(p_v)
        tb = time()
        self.c_factor_tree_data.sum_compute_tree(parameters, ind.cvalues)
        tc = time()
        print 'init: ', tb-ta,', sum_compute: ', tc-tb
    
    def get_factor(self, int id): 
        cdef PyFactor_data factor
        cdef Factor_Node_data node
        if id == -1:
            node= deref(self.c_factor_tree_data.root)
        elif id < self.c_factor_tree_data.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree_data.inner_nodes[id])
        elif id < 2*self.c_factor_tree_data.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree_data.leave_nodes[id-self.c_factor_tree_data.inner_nodes.size()])
        else:
            raise IndexError('id out of bounds')
        factor = PyFactor_data([],[],0)
        factor.c_factor_data = deref(node.factor)
#        factor.c_factor_data = deref(node.factor)
        return factor

    def se_data(self, list cluster, Indicators ind):
        cdef vector[int] cluster_v
        cdef int c
        cdef list 
        cdef vector[double] se
        cdef vector[double].iterator it_prob
#        cdef list l_prob = []
        l_prob = []
        for c in cluster:
            cluster_v.push_back(c)
        
        se = self.c_factor_tree_data.se_data(cluster_v,self.c_factor_tree_data.parameters, ind.cvalues)
        
        it_prob = se.begin()
        while it_prob != se.end():
            l_prob.append(deref(it_prob))
            inc(it_prob)
        return l_prob

    def se_data_parallel(self, list cluster, Indicators_vector ind):
        cdef vector[int] cluster_v
        cdef int c
        cdef list 
        cdef vector[double] se
        cdef vector[double].iterator it_prob
#        cdef list l_prob = []
        l_prob = []
        for c in cluster:
            cluster_v.push_back(c)
        
        se = self.c_factor_tree_data.se_data_parallel(cluster_v,self.c_factor_tree_data.parameters, ind.ind_vector)
        
        it_prob = se.begin()
        while it_prob != se.end():
            l_prob.append(deref(it_prob))
            inc(it_prob)
        return l_prob


    # Compute num_it steps of EM
    def em(self, Indicators ind, int num_it, double alpha):
        cdef vector[double] ll_v
        cdef list ll_l
        cdef int i
        ll_v = self.c_factor_tree_data.em(ind.cvalues, num_it, alpha)
        ll_l = []
        for i in range(ll_v.size()):
            ll_l.append(ll_v[i])
        return ll_v
        
    def em_parallel(self, Indicators_vector ind, int num_it, double alpha):
        cdef vector[double] ll_v
        cdef list ll_l
        cdef int i
        ll_v = self.c_factor_tree_data.em_parallel(ind.ind_vector, num_it, alpha)
        ll_l = []
        for i in range(ll_v.size()):
            ll_l.append(ll_v[i])
        return ll_v
        
    
    def get_parameters(self):
        cdef vector[double] p_v
        cdef int i,j, num_nds
        cdef list param_list, param_xi
        num_nds = self.c_factor_tree_data.parameters.size()        
        param_list = []
        
        for i in range(num_nds):
            p_v = self.c_factor_tree_data.parameters[i]
            param_xi = []
            for j in range(p_v.size()):
                param_xi.append(p_v[j])
            param_list.append(param_xi)
        return param_list
        
    def set_parameters(self, list nodes, list params):
        cdef vector[vector[double]] param_all
        cdef vector[double] param_vector 
        for node_i, pari in zip(nodes,params):
            param_vector.clear()
            for parij in pari:
                param_vector.push_back(parij)
            param_all.push_back(param_vector)
        self.c_factor_tree_data.parameters = param_all
            
    def learn_parameters(self, datat data, double alpha = 0.0):
        self.c_factor_tree_data.learn_params(data.dat_c, data.nrows, data.ncols, alpha)   
    
    def learn_parameters_vars(self, datat data, list variables, double alpha = 0.0):
        for xi in variables:
            self.c_factor_tree_data.learn_params(data.dat_c, data.nrows, data.ncols, alpha, xi)
        
    def log_likelihood(self, Indicators ind):
        return self.c_factor_tree_data.log_likelihood(ind.cvalues)
    
    def log_likelihood_parallel(self, Indicators_vector ind):
        return self.c_factor_tree_data.log_likelihood_parallel(ind.ind_vector)
        
    def copy_parameters(self, PyFactorTree_data ftd, list nodes):
        cdef int xi
        for xi in nodes:
            self.c_factor_tree_data.parameters[xi] = ftd.c_factor_tree_data.parameters[xi]                
        
    def em_parameters(self, int xi, PyFactorTree_data ftd, Indicators ind, int alpha):
        cdef vector[int] cluster_v
        cdef vector[double] se
        #Compute se
        cluster_v = self.c_factor_tree_data.variables[xi]
        #learn parameters from se
        
        se = ftd.c_factor_tree_data.se_data(cluster_v, ftd.c_factor_tree_data.parameters, ind.cvalues)
        return self.c_factor_tree_data.learn_params_se(se, alpha,xi)
        
                
    def em_parameters_parallel(self, int xi, PyFactorTree_data ftd, Indicators_vector ind, int alpha):
        cdef vector[int] cluster_v
        cdef vector[double] se
        #Compute se
        cluster_v = self.c_factor_tree_data.variables[xi]
        #learn parameters from se
        se = ftd.c_factor_tree_data.se_data_parallel(cluster_v, ftd.c_factor_tree_data.parameters, ind.ind_vector)
        return self.c_factor_tree_data.learn_params_se(se, alpha,xi)
        
         
        
