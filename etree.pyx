from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc
cimport numpy as np
import numpy as np
from libcpp cimport bool

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
    
    def __cinit__(self,int n):
        self.c_elimination_tree = Elimination_Tree(n)
    
    def set_parent(self, int xout,int xin, bool inner):
        self.c_elimination_tree.set_parent(xout,xin,inner)
        
    def add_arc(self, int xout,int xin):
        self.c_elimination_tree.add_arc(xout,xin)
    def add_arc_opt(self, int xout,int xin):
        xopt = self.c_elimination_tree.add_arc(xout,xin)
        self.c_elimination_tree.optimize(xopt)
    def remove_arc(self, int xout,int xin):
        self.c_elimination_tree.remove_arc(xout,xin)
    def remove_arc_opt(self, int xout,int xin):
        xopt = self.c_elimination_tree.remove_arc(xout,xin)
        self.c_elimination_tree.optimize(xopt)
    def check_clusters(self):
        return self.c_elimination_tree.check_clusters()
    def swap(self, int xi):
        return self.c_elimination_tree.swap(xi)
        
    def soundness(self):
         return self.c_elimination_tree.soundness()
    def completeness(self):
         return self.c_elimination_tree.completeness()
         
    def push_back_cluster(self, int xi,int x_clust):
        self.c_elimination_tree.cluster_inner.at(xi).push_back(x_clust)
        
    def tw(self):
        return self.c_elimination_tree.tw()

    def size(self):
        return self.c_elimination_tree.size()
    
    
    def get_et_values(self):
        
        cluster_inner = []
        cluster_leaves = []
        children_inner = []
        children_leaves = []
        parents_inner = []
        parents_leaves = []
        for i in range(self.c_elimination_tree.cluster_inner.size()):
            parents_inner.append(self.c_elimination_tree.parents_inner[i])
            parents_leaves.append(self.c_elimination_tree.parents_leaves[i])
            cl_i = self.c_elimination_tree.cluster_inner[i]
            cl_l = self.c_elimination_tree.cluster_leaves[i]
            ch_i = self.c_elimination_tree.children_inner[i]
            ch_l = self.c_elimination_tree.children_leaves[i]
            cluster_inner.append([])
            for j in range(cl_i.size()):
                cluster_inner[i].append(cl_i[j])
            cluster_leaves.append([])
            for j in range(cl_l.size()):
                cluster_leaves[i].append(cl_l[j])
            children_inner.append([])
            for j in range(ch_i.size()):
                children_inner[i].append(ch_i[j])
            children_leaves.append([])
            for j in range(ch_l.size()):
                children_leaves[i].append(ch_l[j])
        return cluster_inner, cluster_leaves, parents_inner, parents_leaves, children_inner, children_leaves
    
    def comp_from_et(self,et):
        for i in range(et.nodes.num_nds):
            parents = et.nodes[i].parents.display()
            var = [i] + parents
            var.sort()
            # Set clusters of leaf nodes
            self.c_elimination_tree.cluster_leaves[i].clear()
            for xi in var:
                self.c_elimination_tree.cluster_leaves[i].push_back(xi)
            if et.nodes[i].parent_et != -1:
                self.c_elimination_tree.set_parent(et.nodes[i].parent_et,i,True)
            self.c_elimination_tree.set_parent(et.nodes[i].nFactor,i,False)
        self.c_elimination_tree.compute_clusters()
    
    def comp_to_et(self,et):
        cdef int i, num_nds, j
        cdef vector[int] var
        num_nds = et.nodes.num_nds
        
        for i in range(num_nds):
            var = self.c_elimination_tree.cluster_leaves[i]
            for j in range(var.size()):
                if var[j]!= i:
                    et.setArcBN_py(var[j],i)
            et.set_arc_et_py(self.c_elimination_tree.parents_inner[i],i)
            et.set_factor_py(i,self.c_elimination_tree.parents_leaves[i])
        et.evaluate()

        
    # Get topological sort. If shallow = T from shallowest to deepest
    def get_topo_sort(self, shallow = True): #TODO
        cdef vector[int] order
        cdef vector[int].iterator it_order
        cdef list order_list
    
        order = self.c_elimination_tree.topo_sort(self.c_elimination_tree.parents_inner, self.c_elimination_tree.children_inner)
        order_list = []
        it_order = order.begin()
        while it_order != order.end():
            order_list.append(deref(it_order))
            inc(it_order)
        if shallow:
            order_list.reverse()
        return order_list
        
        
        
    
        
    
    def compute_clusters(self):
        self.c_elimination_tree.compute_clusters()
        
    
    def compare_et(self, et):
        cluster_inner, cluster_leaves, parents_inner, parents_leaves, children_inner, children_leaves = self.get_et_values()
        n = len(parents_inner)
        cluster_inner_et = [ [i] + et.nodes[i].cluster.display() for i in range(n)]
        cluster_leaves_et = [[i] + et.nodes[i].parents.display() for i in range(n)]
        parents_inner_et = [et.nodes[i].parent_et for i in range(n)]
        parents_leaves_et = [et.nodes[i].nFactor for i in range(n)]
        children_inner_et = [et.nodes[i].children_et.display() for i in range(n)]
        flag = True
        
        if not compare_vectors_of_sets(cluster_inner,cluster_inner_et):
            print
            print 'diff in cluster_inner: '
            print cluster_inner
            print cluster_inner_et
            flag = False
            
        if not compare_vectors_of_sets(cluster_leaves,cluster_leaves_et):
            print
            print 'diff in cluster_leaves: '
            print cluster_leaves
            print cluster_leaves_et        
            flag = False
            
        if not compare_vectors(parents_inner,parents_inner_et):
            print
            print 'diff in parents_inner: '
            print parents_inner
            print parents_inner_et        
            flag = False
            
        if not compare_vectors(parents_leaves,parents_leaves_et):
            print
            print 'diff in parents_leaves: '
            print parents_leaves
            print parents_leaves_et        
            flag = False
            
        if not compare_vectors_of_sets(children_inner,children_inner_et):
            print
            print 'diff in children_inner: '
            print children_inner
            print children_inner_et        
            flag = False
        
        return flag
        
        
def compare_vectors(v1, v2):
    for x1,x2 in zip(v1,v2):
        if x1!=x2:
            return False
    return True
        
def compare_vectors_of_sets(v1, v2):
    for x1,x2 in zip(v1,v2):
        if set(x1)!=set(x2):
            return False
    return True
        
    
