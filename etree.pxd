from libcpp cimport bool
from libcpp.vector cimport vector
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
        vector[int] topo_sort(vector[int] parents_inner, vector[vector[int]] children_inner);



        
        vector[vector[int]] cluster_inner, cluster_leaves, children_inner, children_leaves
        vector[int] parents_inner, parents_leaves

cdef class PyEliminationTree:
    cdef Elimination_Tree c_elimination_tree
    
    