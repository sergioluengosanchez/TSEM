import numpy as np
from cpython cimport bool
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from time import time

#import bnet

#---------------------PT node-----------------------------#

cdef class NodeET(object): 
    
    cdef public Node_Set parents
    cdef public Node_Set children
    cdef public Node_Set children_et
    cdef public Node_Set factors
    cdef public int ncls
    cdef public list classes
    cdef public int nFactor
    cdef public int parent_et
    cdef public str name
    cdef public Node_Set cluster
    cdef public long evaluation
    
    
    def __init__(self,str name = '',object classes = [],int num_nds = 0): 
        if num_nds == 0:
            return
        self.name = name
        self.parent_et = -1
        self.children_et = Node_Set(num_nds)
        self.children = Node_Set(num_nds)
        self.parents = Node_Set(num_nds)
        self.factors = Node_Set(num_nds)
        self.ncls = len(classes)
        self.classes = classes
        self.nFactor = -1
        self.cluster = Node_Set(num_nds)
        self.evaluation = 0
    
    def __cinit__(self):
        return
    
    cdef public NodeET copy(self): 
        cdef NodeET nd = NodeET.__new__(NodeET)
        nd.name = self.name
        nd.parent_et = self.parent_et 
        nd.children_et = self.children_et.copy()
        nd.children = self.children.copy()
        nd.parents = self.parents.copy()
        nd.factors = self.factors.copy()
        nd.ncls = self.ncls
        nd.classes = self.classes
        nd.nFactor = self.nFactor
        nd.cluster = self.cluster.copy()
        nd.evaluation = self.evaluation
        return nd

        


#---------------------list of PT nodes-----------------------------#

cdef class NodeET_lst(object): 
    cdef public NodeET root
    cdef public np.ndarray nds
    cdef public int num_nds
    
    def __init__(self, object varNames = [],object varClasses = [], int num_nds = 0): 
        cdef list lst_nds = []
        if num_nds == 0:
            return
        self.root = NodeET('*',[],num_nds)
        for i in xrange(num_nds):
            lst_nds.append(NodeET(varNames[i],varClasses[i],num_nds))
            self.root.children_et.add(i)
        self.root.parent_et = -1
        self.nds = np.array(lst_nds, dtype=NodeET)
        self.num_nds = num_nds
    
    def __cinit__(self):
        return
    
    def __getitem__(self,int index): 

        if index == -1:
            return self.root
        elif index <-1 or index >= self.num_nds:
            raise IndexError('get: value out of index')
        return self.nds[index]
    
    cdef NodeET get(self,int index): 
        if index == -1:
            return self.root
        elif index <-1 or index >= self.num_nds:
            raise IndexError('get: value out of index')
        return self.nds[index]
    
    cdef set(self,int index,NodeET nd):
        if index == -1:
            self.root = nd
            return
        elif index <-1 or index >= self.num_nds:
            raise IndexError('set: value out of index')
        self.nds[index] = NodeET()
        self.nds[index] = nd

    cdef int size(self): 
        return self.num_nds
    
    cdef NodeET_lst copy(self): 
        cdef NodeET_lst ndl = NodeET_lst.__new__(NodeET_lst)
        cdef NodeET aux
        cdef int i
        cdef list lst_nds
        lst_nds = []
        ndl.nds = np.array([], dtype=NodeET)
        ndl.root = self.root.copy()
        for i in xrange(self.num_nds):
            aux = self.nds[i]
            lst_nds.append(aux.copy())
        ndl.nds = np.array(lst_nds, dtype=NodeET)
        ndl.num_nds = self.num_nds
        
        return ndl
    
    def copy_py(self): 
        cdef NodeET_lst ndl = NodeET_lst()
        cdef NodeET aux
        cdef int i
        cdef list lst_nds
        lst_nds = []
        ndl.nds = np.array([], dtype=NodeET)
        ndl.root = self.root.copy()
        for i in xrange(self.num_nds):
            aux = self.nds[i]
            lst_nds.append(aux.copy())
        ndl.nds = np.array(lst_nds, dtype=NodeET)
        ndl.num_nds = self.num_nds
        
        return ndl


cdef class ElimTree(object):
    
    cdef public str bnName
    cdef public NodeET_lst nodes
    
    #--------------------------------Constructors and destructors------------------------------------------#
    
    #Recieves the name and a list of the classes of each variable of the network
    def __init__(self, str bnName = '', object varNames = [], object varClasses = []):
        cdef int num_nds,xi
        
        if len(varNames) == 0:
            return
        self.bnName = bnName
        num_nds = len(varNames)
        self.nodes = NodeET_lst(varNames,varClasses,num_nds)
        for xi in xrange(self.nodes.size()):
            self.set_factor(xi,xi)
        
    
    def __cinit__(self):
        return
    
    
    def __reduce__(self):
        cdef object varNames,varClasses,state
        cdef int i
        cdef NodeET ndi
        cdef Node_Set cluster
        
        varNames = []
        varClasses = []
        state = []
        for i in xrange(self.nodes.size()):
            ndi = self.nodes[i]
            varNames.append(ndi.name)
            varClasses.append(ndi.classes)
            state.append([])
            state[i].append(ndi.parent_et)
            state[i].append(ndi.children_et.display())
            state[i].append(ndi.parents.display())
            state[i].append(ndi.factors.display())
            state[i].append(ndi.children.display())
            state[i].append(ndi.ncls)
            state[i].append(ndi.nFactor)
            state[i].append(ndi.evaluation)
            state[i].append(ndi.cluster.display())
            
        
        #root
        i = -1
        ndi = self.nodes[i]
        state.append([])
        state[i].append(ndi.parent_et)
        state[i].append(ndi.children_et.display())
        state[i].append(ndi.parents.display())
        state[i].append(ndi.factors.display())
        state[i].append(ndi.children.display())
        state[i].append(ndi.ncls)
        state[i].append(ndi.nFactor)
        state[i].append(ndi.evaluation)
        state[i].append([])
        
        
        return(self.__class__,(self.bnName,varNames,varClasses),state)
    
    def __setstate__(self,state):
        cdef int i,i2
        cdef object sti
        cdef int stij
        cdef NodeET ndi
        for i,sti in enumerate(state):
            i2 = i
            if i == self.nodes.size():
                i2 = -1
            ndi = self.nodes[i2]
            ndi.parent_et= sti[0]
            ndi.children_et = Node_Set(self.nodes.size())
            for stij in sti[1]:
                ndi.children_et.add(stij)
            ndi.parents = Node_Set(self.nodes.size())
            for stij in sti[2]:
                ndi.parents.add(stij)
            ndi.factors = Node_Set(self.nodes.size())
            for stij in sti[3]:
                ndi.factors.add(stij)
            ndi.children = Node_Set(self.nodes.size())
            for stij in sti[4]:
                ndi.children.add(stij)
            ndi.ncls = sti[5]
            ndi.nFactor = sti[6]
            ndi.evaluation = sti[7]
            ndi.cluster = Node_Set(self.nodes.size())
            for stij in sti[8]:
                ndi.cluster.add(stij)
    
    
    #--------------------------------         Compilation         ------------------------------------------
    
    # Compiles an elimination orderion
    def compile_ordering(self, list order):
        cdef list factor_assigned
        cdef int i, j, xi, xj
        cdef Node_Set visited, eval_nodes, available, available_aux
        cdef NodeET node_xi, node_xj
        
        factor_assigned = [False for _ in range(len(order))]
        available = Node_Set(self.nodes.num_nds)
        
        eval_nodes = Node_Set(self.nodes.num_nds)
        for xi in order:
            self.set_arc_et(-1,xi)
            self.set_factor(xi,xi)
            eval_nodes.add(xi)
        self.evaluate_nodes(eval_nodes)
        
        for xi in order:
            # Search for possible parents
            available_aux = available.copy()
            for j in range(available_aux.len()):
                xj = available_aux[j]
                node_xj = self.nodes[xj]
                if node_xj.cluster.contains(xi):
                    self.set_arc_et(xi,xj)
                    available.remove(xj)
            # Search for unassigned factors
            for xj in range(len(factor_assigned)):
                node_xj = self.nodes[xj]
                if factor_assigned[xj] == False and (xi == xj or node_xj.parents.contains(xi)):
                    self.set_factor(xj,xi)
                    factor_assigned[xj] = True
            eval_nodes = Node_Set(self.nodes.num_nds)
            eval_nodes.add(xi) 
            self.evaluate_nodes(eval_nodes)
            available.add(xi)
        
        
    
    #Sets the new parents and children in a PT of an arc, removing old children
    cdef set_arc_et(self,int xout,int xin):
        cdef int parentXin
        cdef NodeET nptaux
        parentXin = self.nodes.get(xin).parent_et
        nptaux = self.nodes.get(parentXin)
        nptaux.children_et.remove(xin)
        nptaux = self.nodes.get(xout)
        nptaux.children_et.add(xin)
        nptaux = self.nodes.get(xin)
        nptaux.parent_et = xout
        nptaux = self.nodes.get(parentXin)
        
    def set_arc_et_py(self,int xout,int xin):
        cdef int parentXin
        cdef NodeET nptaux
        parentXin = self.nodes.get(xin).parent_et
        nptaux = self.nodes.get(parentXin)
        nptaux.children_et.remove(xin)
        nptaux = self.nodes.get(xout)
        nptaux.children_et.add(xin)
        nptaux = self.nodes.get(xin)
        nptaux.parent_et = xout
        nptaux = self.nodes.get(parentXin)
    
    #Sets the new parents and children in a BN of an arc
    cdef setArcBN(self,int xout,int xin):
        cdef NodeET nptaux
        nptaux = self.nodes[xout]
        nptaux.children.add(xin)
        nptaux = self.nodes[xin]
        nptaux.parents.add(xout)
    
    def setArcBN_py(self,int xout,int xin):
        cdef NodeET nptaux
        nptaux = self.nodes[xout]
        nptaux.children.add(xin)
        nptaux = self.nodes[xin]
        nptaux.parents.add(xout)
    
    #Sets the new parents and children in a BN of an arc
    cdef removeArcBN(self,int xout,int xin):
        cdef NodeET nptaux
        nptaux = self.nodes[xout]
        nptaux.children.remove(xin)
        nptaux = self.nodes[xin]
        nptaux.parents.remove(xout)
        
    def removeArcBN_py(self,int xout,int xin):
        cdef NodeET nptaux
        nptaux = self.nodes[xout]
        nptaux.children.remove(xin)
        nptaux = self.nodes[xin]
        nptaux.parents.remove(xout)


    #Sets the factor of node xi to xnew
    cdef set_factor(self,int xi,int xnew):
        cdef NodeET nptaux
        cdef int old
        old = self.nodes[xi].nFactor
        if old != -1:
            nptaux = self.nodes[old]
            nptaux.factors.remove(xi)
        nptaux = self.nodes[xnew]
        nptaux.factors.add(xi)
        self.nodes[xi].nFactor = xnew
    
    
    def set_factor_py(self,int xi,int xnew):
        cdef NodeET nptaux
        cdef int old
        old = self.nodes[xi].nFactor
        if old != -1:
            nptaux = self.nodes[old]
            nptaux.factors.remove(xi)
        nptaux = self.nodes[xnew]
        nptaux.factors.add(xi)
        self.nodes[xi].nFactor = xnew
        
    # Returns the nodes to split and optimize Xopt
    def add_arc(self,int xout,int xin):
        cdef int xf,xk,xk2,i,pXout
        cdef Node_Set predPTFin,predPTXout,Xopt,predaux
        cdef ElimTree fpt_aux

        #t1 = time()
        predaux = Node_Set.__new__(Node_Set,self.nodes.size())
        xf = self.nodes.get(xin).nFactor
        self.setArcBN(xout,xin)
        predPTFin = self.getPred(xf)
        predPTFin.add(xf)
        predPTXout = self.getPred(xout)
        #t2 = time()

        if predPTFin.contains(xout):
            predaux = self.getPred(xf)
            predaux.add(xf)
            #t3 = time()
            Xopt = self.getDesc(xout).intersection(predaux)
            Xopt.add(xout)
#            print 'a'
            #print 'a. t1 = ', t2 - t1, ', t2 = ', t3 - t2, ', t3 = ', time() - t2
            self.evaluate_nodes(Xopt)            
            return Xopt
        elif predPTXout.contains(xf):
            predaux = self.getPred(xout)
            predaux.add(xout)
            #t3 = time()
            Xopt = self.getDesc(xf).intersection(predaux)
            self.set_factor(xin,xout)
            #print 'b. t1 = ', t2 - t1, ', t2 = ', t3 - t2, ', t3 = ', time() - t2
#            print 'b'
            self.evaluate_nodes(Xopt)
            return Xopt
        else:
            #t1 = time()
            fpt_aux = self.copyInfo_cy()
            #t2 = time()
            xk = xf
            for i in xrange(predPTFin.len()):
                pXout = predPTFin.get(i)
                if not predPTXout.contains(pXout):
                    xk = pXout
                    break
            xm = self.nodes.get(xk).parent_et #CF
            self.set_arc_et(xout,xk)
            xk2 = xout
            for i in xrange(predPTXout.len()):
                pXout = predPTXout.get(i)
                if not predPTFin.contains(pXout):
                    xk2 = pXout
                    break
            fpt_aux.set_arc_et(xf,xk2)
            fpt_aux.set_factor(xin,xout)

            predaux = self.getPred(xf)
            predaux.add(xf)
            Xopt = self.getDesc(xm).intersection(predaux) 

            fpt_aux.evaluate_nodes(Xopt)
            self.evaluate_nodes(Xopt)
            
            if fpt_aux.tw(False) < self.tw(False) or (fpt_aux.tw(False) == self.tw(False) and fpt_aux.size() < self.size()): 
                self.nodes = fpt_aux.nodes
            
            #t4 = time()
            #print 'c. t1 = ', t2 - t1, ', t2 = ', t3 - t2, ', t3 = ', t4 - t3, ', t4 = ', time() - t4
#            print 'c'
            return Xopt
    
    
    ## Compiles arc removal xout->xin
    def remove_arc(self,int xout,int xin):  ###CHECK###
        cdef int xf,xj,xi,xip,xjp
        cdef NodeET ndj,ndi
        cdef Node_Set predPTFin,nds_eval,nds_opt
        
        nds_opt = Node_Set.__new__(Node_Set,self.nodes.size())
        nds_eval = Node_Set.__new__(Node_Set,self.nodes.size())
        xf = self.nodes.get(xin).nFactor
        self.removeArcBN(xout,xin)
        self.evaluate() #Update clusters
        
        xj = xout
        ndj = self.nodes.get(xj)
        predPTFin = self.getPred(xf) 
        predPTFin.add(xf)
        if self.nodes.get(xin).nFactor == xout:
            ndf = self.nodes.get(xin)
            cluster = ndf.parents.copy() 
            cluster.add(xin) 
            while xj != -1 and not cluster.contains(xj):
                nds_eval.add(xj)
                nds_opt.add(xj)
                xi = xj
                xj = ndj.parent_et
                ndj = self.nodes.get(xj)
            ndi = self.nodes.get(xi)
            self.set_factor(xin,xj)
            self.evaluate_nodes(nds_eval,False)
        else:
            xi = -1
            for i in xrange(ndj.children_et.len()):
                xi = ndj.children_et.get(i)
                if predPTFin.contains(xi):
                    break
            ndi = self.nodes.get(xi)
        while not ndi.cluster.contains(xj):
            # let xjp be the deepest node in cluster(xi)
            xjp = xj
            nds_eval.clear()
            while xjp != -1 and not ndi.cluster.contains(xjp):
                nds_eval.add(xjp)
                nds_opt.add(xjp)
                xip = xjp
                xjp = self.nodes.get(xip).parent_et
            if xjp == xj:
                return nds_opt
            if xjp!= -1:
                nds_opt.add(xjp)
                nds_eval.add(xjp)
            self.set_arc_et(xjp,xi)
            
            #Update for new iteration
            xi = xip
            xj = xjp
            ndj = self.nodes.get(xj)
            ndi = self.nodes.get(xi)
            self.evaluate_nodes(nds_eval,False)

        return nds_opt

    
            
    
    
    #--------------------------------         Evaluation         ------------------------------------------
    #Evaluate the complexity of node xi in PT
    cdef Node_Set evaluate_node(self,int xi,long *result): 
        #cdef float evalt, evalt2,cmplx0,cmplx,result1
        cdef long evalt, evalt2,cmplx0,cmplx,result1
        cdef int nfactors,xj,lvars,j,k,xk
        cdef NodeET ndi,ndj
        cdef Node_Set nds,nds2
        
        lvars = self.nodes.size()
        evalt = 0
        ndi = self.nodes.get(xi)
        nfactors = 0
        nds = Node_Set.__new__(Node_Set,lvars)
        #1. Obtain nodes in factor lists hanging from xi
        for j in xrange(ndi.factors.len()):
            xj = ndi.factors.get(j)
            ndj = self.nodes.get(xj)
            nds.add(xj)
            for k in xrange(ndj.parents.len()):
                xk = ndj.parents.get(k)
                nds.add(xk)
            nfactors += 1
        #2. Obtain eval and nodes of the factors of each children of Xi
        for j in xrange(ndi.children_et.len()):
            xj = ndi.children_et.get(j)
            nds2 = self.evaluate_node(xj,&evalt2)
            evalt += evalt2
            nds = nds.union(nds2)
            nfactors += 1
        nds.remove(xi)
        cmplx0 = 1
        for j in xrange(nds.len()):
            xj = nds.get(j)
            ndj = self.nodes.get(xj)
            cmplx0 *= ndj.ncls
        cmplx = cmplx0 * ((ndi.ncls * (nfactors)) + (ndi.ncls))
        result1 = cmplx + evalt
        ndi.evaluation = cmplx
        ndi.cluster = nds.copy()
        result[0] = result1
        return nds
    
    #Evaluate nodes. If sort_nodes = False, nodes should be currently ordered from deepest to shallowest in nds:
    cdef long evaluate_nodes(self,Node_Set nds, bool sort_nodes=True): 
        cdef int i,xi,j,xj,k,xk
        cdef NodeET ndi,ndj
        cdef Node_Set cluster, nodes_topo_sort, nodes_topo_sort_aux
        cdef long result,cmplx,cmplx0,nfactors
        
        
        if sort_nodes:
            nodes_topo_sort = Node_Set.__new__(Node_Set,self.nodes.size())
            nodes_topo_sort_aux= self.orderByDepth(nds)
            for i in range(nodes_topo_sort_aux.len()):
                xi = nodes_topo_sort_aux.get(nodes_topo_sort_aux.len()-i-1)
                nodes_topo_sort.add(xi)
        else:
            nodes_topo_sort = nds
        
        for i in range(nodes_topo_sort.len()):
            xi = nodes_topo_sort.get(i)
#            print
#            print 'xi: ', xi
            
            nfactors = 0
            ndi = self.nodes.get(xi)
            cluster = ndi.cluster
            cluster.clear()

            for j in xrange(ndi.factors.len()):
                xj = ndi.factors.get(j)
                ndj = self.nodes.get(xj)
#                print 'fac xj: ', xj, ndj.parents.display()
                cluster.add(xj)
                for k in xrange(ndj.parents.len()):
                    xk = ndj.parents.get(k)
                    cluster.add(xk)
                nfactors += 1
            
            for j in xrange(ndi.children_et.len()):
                xj = ndi.children_et.get(j)
                ndj = self.nodes.get(xj)
#                print 'ch xj: ', xj, ndj.cluster.display()
                cluster = cluster.union(ndj.cluster)
                nfactors += 1
            cluster.remove(xi)
            cmplx0 = 1
            cluster.add(xi)
            cluster.remove(xi)
            for j in range(cluster.len()):
                xj = cluster.get(j)
                ndj = self.nodes.get(xj)
                cmplx0 *= ndj.ncls
            cmplx = cmplx0 * ((ndi.ncls * (nfactors)) + (ndi.ncls))
            
            ndi.evaluation = cmplx
            ndi.cluster = cluster
        
        for i in xrange(self.nodes.size()):
            ndi = self.nodes.get(i)
            result += ndi.evaluation
        return result
        
    #Obtain size from nodes
    def size(self):
        cdef int i
        cdef long complexity
        complexity = 0
        for i in xrange(self.nodes.num_nds):
            complexity += self.nodes[i].evaluation
        return complexity


    #Evaluate the complexity of the PT
    def evaluate(self):
        cdef int xi,i
        cdef long rs = 0
        cdef long rsi = 0
        cdef Node_Set nds
        cdef NodeET nptaux = self.nodes[-1]
        for i in xrange(nptaux.children_et.len()):
            xi = nptaux.children_et[i]
            nds = self.evaluate_node(xi,&rsi)
            rs += rsi
        return rs
    
    #Evaluate the complexity of the PT: In cython
    cdef long evaluate_cy(self): 
        cdef int xi,i
        cdef long rs = 0
        cdef long rsi = 0
        cdef Node_Set nds
        cdef NodeET nptaux = self.nodes[-1]
        for i in xrange(nptaux.children_et.len()):
            xi = nptaux.children_et[i]
            self.evaluate_node(xi,&rsi)
            rs += rsi
        return rs
    
    cdef getVars_xi(self,int xi,np.ndarray vars_all):
        cdef int j,factor,xj
        cdef Node_Set var = Node_Set.__new__(Node_Set,self.nodes.size())
        cdef NodeET ndi = self.nodes.get(xi)
        
        for j in xrange(ndi.factors.len()):
            factor = ndi.factors[j]
            var.add(factor)
            var = var.union(self.nodes[factor].parents)
        
        for j in xrange(ndi.children_et.len()):
            xj = ndi.children_et[j]
            self.getVars_xi(xj,vars_all)
            var = var.union(vars_all[xj])
        vars_all[xi] = var
    
    
    #Returns all the nodes contained in any factor that is 
    cdef np.ndarray getVars(self): #O(n^2)
        cdef np.ndarray vars_all = np.array([Node_Set.__new__(Node_Set,self.nodes.size()) for _ in xrange(self.nodes.size())], dtype=Node_Set)
        cdef int i,xi
        cdef NodeET root
        root = self.nodes.get(-1)
        for i in xrange(root.children_et.len()):
            xi = root.children_et[i]
            self.getVars_xi(xi,vars_all)
        return vars_all
    
    
    #Returns the cluster of each node
    cdef np.ndarray getClusters(self): #O(n^2)
        cdef np.ndarray predsPT = np.array([Node_Set(self.nodes.size()) for _ in xrange(self.nodes.size())], dtype=Node_Set)
        cdef np.ndarray var = self.getVars() #O(n^2)
        cdef np.ndarray clusters = np.array([Node_Set(self.nodes.size()) for _ in xrange(self.nodes.size())], dtype=Node_Set)
        cdef Node_Set predxi, varxi, cluster_xi
        cdef int xi
        
        self.getPredPT_all(predsPT,-1)
        for xi in xrange(self.nodes.size()):
            predxi = predsPT[xi]
            predxi.add(xi)
            varxi = var[xi]
            cluster_xi = varxi.intersection(predxi)
            cluster_xi.remove(xi)
            clusters[xi] = cluster_xi
        return clusters
    
    #Treewidth of the pt. If update= True the clusters are computed, otherwise it is assumed that they are already updated
    def tw(self,update=True):
        cdef np.ndarray clusters
        cdef int tw,tw1,i
        cdef Node_Set cl

        if update:
            clusters = self.getClusters()
            tw = 0
            for i in xrange(len(clusters)):
                cl = clusters[i]
                tw1 = cl.len()
                if tw1 > tw:
                    tw = tw1
        else:
            tw = 0
            for i in range(self.nodes.num_nds):
                cl = self.nodes[i].cluster
                tw1 = cl.len()
                if tw1 > tw:
                    tw = tw1
        return tw

    
    #Xi is soundf if factor(Xi) is a descendant in PT of all the nodes in {Xi} U Pa(B,Xi)
    def is_soundNode(self,int xi):
        cdef int i,j,xj,factor
        cdef Node_Set predXiPT,p_f
        cdef NodeET nptaux,nptaux2
        
        predXiPT = self.getPred(xi)
        predXiPT.add(xi)
        nptaux = self.nodes[xi]
        for i in xrange(nptaux.factors.len()):
            factor = nptaux.factors[i]
            nptaux2 = self.nodes[factor]
            p_f = nptaux2.parents.copy()
            p_f.add(factor)
            for j in xrange(p_f.len()):
                xj = p_f[j]
                if not (predXiPT.contains(xj)):
                    print 'Not sound factor',factor,'xj',xj
                    print 'nFactor ',xi,', predPTfactor ',predXiPT.display()
                    nptaux = self.nodes.get(xj)
                    print 'ch xj ', nptaux.children_et.display()
                    return False,factor,xj
        return True,-10,-10
    
    # Returns True if the FPT is sound and false otherwise
    def is_sound(self):
        cdef int xi
        cdef object sound
        for xi in xrange(self.nodes.size()):
            sound = self.is_soundNode(xi)
            if not sound[0]:
                return False,sound[1],sound[2]
            
        return True,-10,-10
    
    # Returns True if the FPT is sound and false otherwise
    def is_complete(self):
        cdef int xi,xj
        cdef NodeET ndi
        cdef tuple sound
        cdef Node_Set cluster
        for xi in xrange(self.nodes.size()):
            ndi = self.nodes.get(xi)
            for j in xrange(ndi.children_et.len()):
                xj = ndi.children_et.get(j)
                cluster = self.nodes.get(xj).cluster
                if not cluster.contains(xi):
                    return False,xi,xj
            
        return True,-10,-10
    
    def is_valid(self):
        if self.is_complete()[0] and self.is_sound()[0]:
            return True
        return False
        
    #--------------------------------         Optimization         ------------------------------------------

    def push_up_node(self, int xi):
#    cdef public bool push_up_node(self, int xi):
        """ Changes the position of node xi with its parent. It assumes that each node is correctly evaluated
        :param xi: Node to push up (int)
        """
        cdef Node_Set nodes_eval, children_aux, factors_aux
        cdef NodeET node_xi, node_xp, node_xj
        cdef int xp, xpp, xj, j
        
        nodes_eval = Node_Set.__new__(Node_Set,self.nodes.size())        
        # Initialize nodes
        node_xi = self.nodes[xi]
        xp = node_xi.parent_et
        if xp == -1:
            return False
        node_xp = self.nodes[xp]
        xpp = self.nodes[xp].parent_et
        children_aux = node_xi.children_et.copy()
        factors_aux = node_xi.factors.copy()
        for j in range(children_aux.len()):
            xj = children_aux[j]
            node_xj = self.nodes[xj]
            if node_xj.cluster.contains(xp):
                self.set_arc_et(xp,xj)
        for j in range(factors_aux.len()):
            xj = factors_aux[j]
            node_xj = self.nodes[xj]
#            print 'xj ', xj,', cluster: ',node_xj.cluster.display()
            if node_xj.parents.contains(xp) or xj == xp:
                self.set_factor(xj,xp)
        
        xpp = node_xp.parent_et
        self.set_arc_et(xpp,xi)
        self.set_arc_et(xi,xp)
        
        nodes_eval.add(xp)
        nodes_eval.add(xi)
        self.evaluate_nodes(nodes_eval,False)

        return True


    # Local optimization for xi
    def pushUpNode(self,int xi,Node_Set nds_aux):
#    cdef public bool pushUpNode(self,int xi,Node_Set nds_aux):
        cdef NodeET ndi,ndp,nptaux
        cdef int xp,h,xh,p,branch,factor,j,xj,k,xl
        cdef bool flag0,flag
        cdef Node_Set factors_aux,branches,factors,descPTXp
        ndi = self.nodes[xi]
        xp = ndi.parent_et
        if xp == -1:
            return False
        ndp = self.nodes[xp]
        # Update Xi parent
        xl = self.nodes.get(xi).parent_et
        nds_aux.add(xi)
        nds_aux.add(xl)
        nds_aux.add(ndp.parent_et)
        self.set_arc_et(ndp.parent_et,xi)
        # 1. Factors assignement
        # 1a: If Xp in Pred(B,Xp) and P(Xi|..) in Factors_Xi, then Xp --> P(Xi|..)
        flag0 = False
        factors_aux = ndi.factors.copy()
        for h in xrange(factors_aux.len()):
            xh = factors_aux.get(h)
            nptaux = self.nodes.get(xh)
            if nptaux.parents.contains(xp):
                xl = self.nodes.get(xh).nFactor
                nds_aux.add(xl)
                nds_aux.add(xh)
                nds_aux.add(xp)
                self.set_factor(xh,xp)
                
                if xh != xi:
                    flag0=True
        # 1b: If P(Xp|..) in Factors_Xi, then Xp --> P(Xp|..)
        if ndp.nFactor == xi:
            xl = self.nodes.get(xp).nFactor
            nds_aux.add(xl)
            nds_aux.add(xp)
            self.set_factor(xp,xp)
        # 2. Branches that contain a factor of Xk in a node Xj such that Xj in Desc(B,Xp) or Xj in Pred(B,Xp)
        #    will hang from Xp. The rest of the nodes will hang from Xi.
        branches = ndi.children_et.union(ndp.children_et)
        for i in xrange(branches.len()):
            branch = branches.get(i)
            nodesBranch = self.getDesc(branch)
            nodesBranch.add(branch)
            flag = False
            for j in xrange(nodesBranch.len()):
                xj = nodesBranch.get(j)
                nptaux = self.nodes.get(xj)
                for k in xrange(nptaux.factors.len()):
                    factor = nptaux.factors.get(k)
                    if (ndp.children.contains(factor)) or (ndp.nFactor == xj):
                        flag = True
                        break
                if flag:
                    break
            if flag:
                xl = self.nodes.get(branch).parent_et
                nds_aux.add(xl)
                nds_aux.add(branch)
                nds_aux.add(xp)
                self.set_arc_et(xp,branch)
                
            else:
                xl = self.nodes.get(branch).parent_et
                nds_aux.add(xl)
                nds_aux.add(branch)
                nds_aux.add(xi)
                self.set_arc_et(xi,branch)
        # 3. If it exist a node a factor of Xk in a node Xj in {Xp} U Desc(P',Xp) such that Xk in Desc(B,Xi) or 
        #    Xk in Pred(B,Xi), then Xp hangs from Xi. Otherwise Xp maintains its old
        #    parent
        descPTXp = self.getDesc(xp)
        flag = False
        if flag0:
            flag=True
        else:
            descPTXp.add(xp)
            for j in xrange(descPTXp.len()):
                xj = descPTXp.get(j)
                nptaux = self.nodes.get(xj)
                for k in xrange(nptaux.factors.len()):
                    factor = nptaux.factors.get(k)
                    if (ndi.children.contains(factor)) or (ndi.nFactor == xj):
                        flag = True
                        break
                if flag:
                    break
        if flag:
            xl = self.nodes.get(xp).parent_et
            nds_aux.add(xl)
            nds_aux.add(xp)
            nds_aux.add(xi)
            self.set_arc_et(xi,xp)
        return True
    
    
    # Local optimization for xi
    cdef bool pushUpNode_new(self,int xi,Node_Set nds_aux): 
        cdef NodeET ndi,ndj,ndh
        cdef int xj,xh,k
        cdef Node_Set cluster,fndi,chndi
        
        ndi = self.nodes.get(xi)
        xj = ndi.parent_et
        if xj == -1:
            return False
        
        ndj = self.nodes.get(xj)
        xh = ndj.parent_et
        self.set_arc_et(xh,xi)
        self.set_arc_et(xi,xj)
        nds_aux.add(xi)
        nds_aux.add(xj)
        nds_aux.add(xh)
        chndi = ndi.children_et.copy()
        for k in xrange(chndi.len()):
            xk = chndi.get(k)
            cluster = self.nodes.get(xk).cluster
            if cluster.contains(xj):
                self.set_arc_et(xj,xk)
                nds_aux.add(xk)
        
        fndi = ndi.factors.copy()
        for k in xrange(fndi.len()):
            xk = fndi.get(k)
            cluster = self.nodes.get(xk).parents.copy()
            cluster.add(xk)
            if cluster.contains(xj):
                self.set_factor(xk,xj)
                nds_aux.add(xk)
        return True
        
    
    #Chooses
    cdef set_factorFromScratch(self,int xi):
        cdef Node_Set desc
        cdef NodeET nodeAux
        desc = self.getDesc(xi)
        for _ in xrange(desc.len()):
            xj = desc.pop()
            nodeAux = self.nodes[xi]
            if nodeAux.parents.contains(xj):
                self.set_factor(xi,xj)
                return
        self.set_factor(xi,xi)
        

    def optimize_tree_width(self, Node_Set x_opt, optimization_type = 'tw'):
        """ Locally reduce the width of an elimination tree
        """
        cdef int i, xi, xp, new_width, old_width
        #cdef long old_size
        cdef NodeET node_xi
        cdef Node_Set nodes_sorted, nds_aux
        cdef bool flag
        
        # Topological sort of
        nodes_sorted = self.orderByDepth(x_opt)          
   
        # Try to reduce the width of each of them
        for i in range(nodes_sorted.len()):
            flag = True
            xi = nodes_sorted.get(i)
            while flag:  
                node_xi = self.nodes.get(xi)
                xp = node_xi.parent_et 
                if xp == -1:
                    flag = False
                else:
                    old_width = self.nodes.get(xi).cluster.len_py()
                    old_size = self.size()
                    self.push_up_node(xi)
                    new_width = self.nodes.get(xp).cluster.len_py()
                    if optimization_type == 'size':
                        if new_width>old_width or (new_width==old_width and self.size()>old_size):
#                        if self.size()>old_size:
                            self.push_up_node(xp)
                            flag = False
                    elif new_width>old_width:
                        self.push_up_node(xp)
                        flag = False
        return 
    
    #Push up all the nodes in Xopt from the shallowest to the deepest (reducing width)
    def optimize_tw(self,Node_Set Xopt):  
        #Order Xopt by depth
        cdef Node_Set Xopt2,nds_aux,nds_eval
        cdef int predEval,newEval,newEval0
        cdef ElimTree bestInfo
        cdef int i,xi,j,xp,xpp
        cdef object ta,tb,tc,td,tpu,tev,tinfo,tx,tev2
        cdef NodeET nptaux,ndi
        
        nds_aux = Node_Set.__new__(Node_Set,self.nodes.size())
        nds_eval = Node_Set.__new__(Node_Set,self.nodes.size())
        tpu = 0
        tev = 0
        tev2 = 0
        tinfo = 0
        Xopt2 = self.orderByDepth(Xopt)
        
        bestInfo = self.copyInfo_cy()
        
        for i in xrange(Xopt2.len()):
            xi = Xopt2.get(i)
            flag = True
            while(flag):
                #Only the width of the X_i position may change
                predEval = self.nodes.get(xi).cluster.len_py()
                nds_aux.clear()
                xp = self.nodes.get(xi).parent_et
                if xp == -1:
                    break
                flag = self.pushUpNode(xi,nds_aux)
                xpp = self.nodes.get(xi).parent_et
                nds_eval.clear()
                nds_eval.add(xp)
                nds_eval.add(xi)
                newEval = self.evaluate_nodes(nds_eval) #New eval has no purpose in this line, its value is given by next line
                newEval = self.nodes.get(xp).cluster.len_py()
                if predEval >= newEval:
                    for j in xrange(nds_aux.len()):
                        xj = nds_aux.get(j)
                        nptaux = self.nodes.get(xj)
                        bestInfo.nodes.set(xj,nptaux.copy())
                else:
                    for j in xrange(nds_aux.len()):
                        xj = nds_aux.get(j)
                        nptaux = bestInfo.nodes.get(xj)
                        self.nodes.set(xj,nptaux.copy())
                    flag = False
        return
        
    #Push up all the nodes in Xopt from the shallowest to the deepest
    def optimize(self,Node_Set Xopt):  
        #Order Xopt by depth
        cdef Node_Set Xopt2,nds_aux,nds_eval
        cdef int bestEval,newEval,newEval0
        cdef ElimTree bestInfo
        cdef int i,xi,j,xp,xpp
        cdef object ta,tb,tc,td,tpu,tev,tinfo,tx,tev2
        cdef NodeET nptaux,ndi
        
        nds_aux = Node_Set.__new__(Node_Set,self.nodes.size())
        nds_eval = Node_Set.__new__(Node_Set,self.nodes.size())
        tpu = 0
        tev = 0
        tev2 = 0
        tinfo = 0
        Xopt2 = self.orderByDepth(Xopt)
        
        bestEval = self.evaluate_cy()
        bestInfo = self.copyInfo_cy()
        
        for i in xrange(Xopt2.len()):
            xi = Xopt2.get(i)
            flag = True
            while(flag):
                nds_aux.clear()
                xp = self.nodes.get(xi).parent_et
                if xp == -1:
                    break
                flag = self.pushUpNode(xi,nds_aux)
                xpp = self.nodes.get(xi).parent_et
                nds_eval.clear()
                nds_eval.add(xp)
                nds_eval.add(xi)
                nds_eval.add(xpp)
                newEval = self.evaluate_nodes(nds_eval)

                if bestEval > newEval:
                    bestEval = newEval
                    for j in xrange(nds_aux.len()):
                        xj = nds_aux.get(j)
                        nptaux = self.nodes.get(xj)
                        bestInfo.nodes.set(xj,nptaux.copy())
                else:
                    for j in xrange(nds_aux.len()):
                        xj = nds_aux.get(j)
                        nptaux = bestInfo.nodes.get(xj)
                        self.nodes.set(xj,nptaux.copy())
                    flag = False
        return
    
#Push up all the nodes in Xopt from the shallowest to the deepest
    def optimize_tw2(self,Node_Set Xopt):
        #Order Xopt by depth
        cdef Node_Set Xopt2,nds_aux,nds_eval
        cdef int bestEval,newEval,newEval0
        cdef ElimTree bestInfo
        cdef int i,xi,j,xp,xpp
        cdef object ta,tb,tc,td,tpu,tev,tinfo,tx,tev2
        cdef NodeET nptaux,ndi

        nds_aux = Node_Set.__new__(Node_Set,self.nodes.size())
        nds_eval = Node_Set.__new__(Node_Set,self.nodes.size())
        tpu = 0
        tev = 0
        tev2 = 0
        tinfo = 0
        Xopt2 = self.orderByDepth(Xopt)

        bestEval = self.tw(False)
        bestInfo = self.copyInfo_cy()

        for i in xrange(Xopt2.len()):
            xi = Xopt2.get(i)
            flag = True
            while(flag):
                nds_aux.clear()
                xp = self.nodes.get(xi).parent_et
                if xp == -1:
                    break
                flag = self.pushUpNode(xi,nds_aux)
                xpp = self.nodes.get(xi).parent_et
                nds_eval.clear()
                nds_eval.add(xp)
                nds_eval.add(xi)
                nds_eval.add(xpp)
                newEval = self.evaluate_nodes(nds_eval)
                newEval = self.tw(False)

                if bestEval > newEval:
                    bestEval = newEval
                    for j in xrange(nds_aux.len()):
                        xj = nds_aux.get(j)
                        nptaux = self.nodes.get(xj)
                        bestInfo.nodes.set(xj,nptaux.copy())
                else:
                    for j in xrange(nds_aux.len()):
                        xj = nds_aux.get(j)
                        nptaux = bestInfo.nodes.get(xj)
                        self.nodes.set(xj,nptaux.copy())
                    flag = False
        return


    # Greedy optimization of elimination trees using dynamic programming
    def optimize_dp(self, str optimization_type = 'width'):
      cdef bool improve = True
      cdef Node_Set swap_diff # Size difference of each node i with its parent  
      cdef NodeET node_xi, node_xp, node_x_best_swap
      cdef Node_Set nodes_update
      cdef int i, xi, x_best_swap, xp
      cdef long best_swap_cmplx
      cdef list et1, et2
      
#      print
#      print 'tw: ', self.tw()
      
      #Initialize swap_diff
      swap_diff = Node_Set.__new__(Node_Set,self.nodes.num_nds) 
      swap_diff.icols = self.nodes.num_nds
      nodes_update = Node_Set.__new__(Node_Set,self.nodes.num_nds) 
      et1 = [self.nodes[i].parent_et for i in range(self.nodes.num_nds)]
      for xi in range(self.nodes.num_nds):
        swap_diff.set(xi, self.score_swap(xi, optimization_type))
#        if swap_diff.get(xi) <= -2147483648:
#            raise ValueError("El imperialihmo yanki")
#        if swap_diff.get(xi) < 0:
#            improve = True
      et2 = [self.nodes[i].parent_et for i in range(self.nodes.num_nds)]

      
#      print 'swap_diff: ', swap_diff.display()     
      #Apply best change until convergence
      while improve:
        # Choose best swap
        best_swap_cmplx = 0 
        for xi in range(swap_diff.len()):
          if swap_diff.get(xi) < best_swap_cmplx:
            best_swap_cmplx = swap_diff.get(xi)
            x_best_swap = xi
#        print 'x_best_swap: ', x_best_swap
#        print 'best_swap_cmplx: ', best_swap_cmplx
        
        # Update swap_diff
        if best_swap_cmplx < 0:
          node_x_best_swap = self.nodes.get(x_best_swap)
          xp = node_x_best_swap.parent_et
          self.push_up_node(x_best_swap) 
          node_xp = self.nodes.get(xp)
          nodes_update = nodes_update.union(node_xp.children_et)
          nodes_update = nodes_update.union(node_x_best_swap.children_et)
          nodes_update.add(x_best_swap)
          nodes_update.display()
#          print 'nd_up 1: ', nodes_update.display()
          et1 = [self.nodes[i].parent_et for i in range(self.nodes.num_nds)]
          for i in range(nodes_update.len()):
            swap_diff.set(nodes_update.get(i), self.score_swap(nodes_update.get(i), optimization_type))
#            print 'diff sd: ', swap_diff.get(nodes_update.get(i))
          nodes_update.clear()
          #print 'nd_up 2: ', nodes_update.display()
          et2 = [self.nodes[i].parent_et for i in range(self.nodes.num_nds)]
        else:
          improve = False
#      print 'final odt'
#      print 'tw2: ', self.tw()
#      print


          
          
        
    # Obtains the size or max width of the nodes involved in the swap of i
    # TODO: Optimizar width del ET en global en vez de en los nodos
    cdef long score_swap(self, int xi, str optimization_type = 'width'):
      cdef NodeET node_xi, node_xp
      cdef int xp
      cdef long old_cmplx, new_cmplx
      cdef long diff
    
      node_xi = self.nodes.get(xi)
      xp = node_xi.parent_et
      node_xp = self.nodes.get(xp)
    
      # Obtain old width or size
      if optimization_type == 'width':
          old_cmplx = node_xi.cluster.len()
      elif optimization_type == 'size':
          old_cmplx = node_xi.evaluation + node_xp.evaluation
      else:
        raise ValueError(optimization_type + ' is not a supported optimization type')
        
      # Swap xi and xp
#      print
#      print 'xi: ', xi
#      print '1. Pa(xi): ', node_xi.evaluation
#      print '1. Pa(xp): ', node_xp.evaluation
#      print '1. Ch(xi): ', node_xi.children_et.display()
#      print '1. Ch(xp): ', node_xp.children_et.display()
#      print '1. cl(xi): ', node_xi.cluster.display()
#      print '1. cl(xp): ', node_xp.cluster.display()
#      print '1. cl_Ch(xi): ', [self.nodes.get(ci).cluster.display() for ci in node_xi.children_et.display()]
#      print '1. cl_Ch(xp): ', [self.nodes.get(ci).cluster.display() for ci in node_xp.children_et.display()]
      self.push_up_node(xi)  
#      print '2. Pa(xi): ', node_xi.evaluation
#      print '2. Pa(xp): ', node_xp.evaluation
#      print '2. Ch(xi): ', node_xi.children_et.display()
#      print '2. Ch(xp): ', node_xp.children_et.display()
#      print '2. cl(xi): ', node_xi.cluster.display()
#      print '2. cl(xp): ', node_xp.cluster.display()
#      print '2. cl_Ch(xi): ', [self.nodes.get(ci).cluster.display() for ci in node_xi.children_et.display()]
#      print '2. cl_Ch(xp): ', [self.nodes.get(ci).cluster.display() for ci in node_xp.children_et.display()]
      
      #TODO: Comprobar que node_xi ha actualizado sus valores de evaluation despues del push up
      if optimization_type == 'width':
          new_cmplx = node_xp.cluster.len()
      elif optimization_type == 'size':
          new_cmplx = node_xi.evaluation + node_xp.evaluation
      else:
          raise ValueError(optimization_type + 'is not a supported optimization type')
#      print 'old: ', old_cmplx, ', new: ', new_cmplx,', diff: ', new_cmplx - old_cmplx
      
      # Swap xp and xi
      self.push_up_node(xp) 
      diff = new_cmplx - old_cmplx
      if diff >= 0:
          return 0
      return -np.log(-diff)



    #Auxiliar functions for reestoring the data of the network
    def copyInfo(self): 
        cdef ElimTree fpt = ElimTree.__new__(ElimTree, 'wheat')
        fpt.bnName = self.bnName
        fpt.nodes = self.nodes.copy()
        return fpt
    
    cdef ElimTree copyInfo_cy(self): 
        cdef ElimTree fpt = ElimTree.__new__(ElimTree, 'wheat')
        fpt.bnName = self.bnName
        fpt.nodes = self.nodes.copy()
        return fpt
        
    def loadInfo(self,ElimTree fpt): 
        self.bnName = fpt.bnName
        self.nodes = fpt.nodes.copy()
        return
    
    cdef loadInfo_cy(self,ElimTree fpt): 
        self.bnName = fpt.bnName
        self.nodes = fpt.nodes.copy()
        return
    
    
    # def plotFPT(self,numbers = True,fileName=None): 
    #     cdef NodeET ndi
    #     nodes = [str(xi) if numbers else self.nodes[xi].name for xi in xrange(self.nodes.size())]
    #     nodes.append('*')
    #     parents = []
    #     for xi in xrange(self.nodes.size()):
    #         ndi = self.nodes[xi]
    #         parentsi = ''
    #         for j in xrange(ndi.parents.len()):
    #             parent = ndi.parents[j]
    #             if numbers:
    #                 parentsi = parentsi + str(parent) + ','
    #             else:
    #                 parentsi = parentsi + self.nodes[parent].name + ','
    #         parents.append(parentsi[:-1])
    #     factors = [str(xi) +'|' + parents[xi] if numbers else self.nodes[xi].name +'|' + parents[xi] for xi in range(self.nodes.size())]
    #
    #     edges = []
    #     for xi in xrange(self.nodes.size()):
    #         ndi = self.nodes[xi]
    #         xj = ndi.parent_et
    #         edges.append((nodes[xj],nodes[xi]))
    #         for j in xrange(ndi.factors.len()):
    #             factor = ndi.factors[j]
    #             edges.append((nodes[xi],factors[factor]))
    #
    #     G = nx.Graph()
    #     G.add_nodes_from(nodes + factors)
    #     G.add_edges_from(edges)
    #     colors1 = [.1 for node in nodes]
    #     colors2 = [.1 for node in factors]
    #
    #     position = nx.graphviz_layout(G)
    #
    #     nx.draw_networkx_nodes(G,position, nodelist=nodes,node_size=[1000 for x in G], node_shape="o",node_color = colors1,cmap=pylab.cm.Blues)
    #     nx.draw_networkx_nodes(G,position, nodelist=factors, node_size=[1000 for x in G],node_shape="s",node_color = colors2,cmap=pylab.cm.Greys)
    #
    #     nx.draw_networkx_edges(G,position)
    #     nx.draw_networkx_labels(G,position)
    #
    #     if fileName != None:
    #         pylab.savefig(fileName)
    #         pylab.close()
    #     else:
    #         pylab.show()
    #
    #     return
    
    cdef getPred_aux(self,int xi,Node_Set pred):
        if self.nodes[xi].parent_et != -1:
            self.getPred_aux(self.nodes[xi].parent_et,pred)
        pred.add(self.nodes[xi].parent_et)
    
    # Returns the predecessors of the input node n1, from the shallowest to the deepest
    cdef Node_Set getPred(self,int xi):
        cdef Node_Set pred
        pred = Node_Set.__new__(Node_Set,self.nodes.size())
        self.getPred_aux(xi,pred)
        return pred
    
    
    cdef getPredPT_all(self,np.ndarray predPT,int xi = -1): #O(n)
        cdef Node_Set nds,ndsaux1
        cdef NodeET ndi
        cdef int parent_et
        
        ndi = self.nodes[xi]
        if xi == -1:
            nds = Node_Set.__new__(Node_Set,self.nodes.size())
            predPT[xi] = nds
        else:
            parent_et = ndi.parent_et
            if parent_et == -1:
                nds = Node_Set.__new__(Node_Set,self.nodes.size())
            else:
                ndsaux1 = predPT[parent_et]
                nds =ndsaux1.copy()
            nds.add(parent_et)
            predPT[xi] = nds
        
        for j in xrange(ndi.children_et.len()):
            xj = ndi.children_et.get(j)
            self.getPredPT_all(predPT,xj)
    
    
    cdef getDesc_aux(self,int xi,Node_Set desc): ##CH##
        cdef int j,xj
        cdef NodeET nptaux
        nptaux = self.nodes[xi]

        for j in xrange(nptaux.children_et.len()):
            xj = nptaux.children_et.get(j)
            desc.add(xj)
            self.getDesc_aux(xj,desc)
    
    # Returns the descendats of the input node n1, from the shallowest to the deepest
    cdef Node_Set getDesc(self,int xi): ##CH##
        cdef Node_Set desc
        desc = Node_Set.__new__(Node_Set,self.nodes.size())
        self.getDesc_aux(xi,desc)
        return desc
    
    cdef Node_Set orderByDepth(self,Node_Set nds): ##CH##
        cdef Node_Set nds2,nds3,nds4
        cdef int xi,j,xj
        cdef NodeET nptaux
        
        nds2 = Node_Set.__new__(Node_Set,self.nodes.size())
        nds3 = Node_Set.__new__(Node_Set,self.nodes.size())
        nds2.add(-1)
        
        while nds2.len()>0:
            nds4 = Node_Set.__new__(Node_Set,self.nodes.size())
            while nds2.len()>0:
                xi = nds2.pop()
                if nds.contains(xi):
                    nds3.add(xi)
                nptaux = self.nodes[xi]
                for j in xrange(nptaux.children_et.len()):
                    xj = nptaux.children_et.get(j)
                    nds4.add(xj)
            nds2 = nds4
        
        return nds3

    def get_preds_bn(self, i):
        ni = self.nodes[i]
        pred = set(ni.parents.display())
        pred_aux = set(ni.parents.display())
        while len(pred_aux) > 0:
            xj = pred_aux.pop()
            nj = self.nodes[xj]
            for xp in nj.parents.display():
                if not (xp in pred):
                    pred.add(xp)
                    pred_aux.add(xp)
        return pred
    
    def get_desc_bn(self, i):
        ni = self.nodes[i]
        desc = set(ni.children.display())
        desc_aux = set(ni.children.display())
        while len(desc_aux) > 0:
            xj = desc_aux.pop()
            nj = self.nodes[xj]
            for xp in nj.children.display():
                if not (xp in desc):
                    desc.add(xp)
                    desc_aux.add(xp)
        return desc

    #Checks if node i is predecessor of node j in the bn
    def isAnt(self, int i,int j):
        return i in self.get_preds_bn(j)
        
    #Return all possible new parents for the variable i
    def predec(self,i):
        cdef list pred = []
        cdef NodeET ni = self.nodes[i]
        lenNodes = self.nodes.size()
        for j in xrange(0,lenNodes):
            if i!=j and not self.isAnt(i,j) and (not ni.parents.contains(j)):
                pred.append(j)
        return pred


cdef class Node_Set(object): ##CH##
    cdef int *dat
    cdef int *dat_bool
#    cdef int dat[2000]
#    cdef int dat_bool[2000]
    cdef int ncols
    cdef int icols
    cdef str name

    def __init__(self,int ncols): ##CH##
        return

    def __cinit__(self,int ncols):
        self.ncols = ncols
        self.dat = <int *>PyMem_Malloc((ncols + 1) * sizeof(int))
        if not self.dat:
            raise MemoryError()
        self.dat_bool = <int *>PyMem_Malloc((ncols + 1) * sizeof(int))
        if not self.dat_bool:
            raise MemoryError()
        self.icols = 0
        for j in xrange(ncols+1):
            self.dat[j] = 0
            self.dat_bool[j] = -1

    def __dealloc__(self): ##CH##'
        
        PyMem_Free(self.dat)
        PyMem_Free(self.dat_bool)

    def __getitem__(self,int index): ##CH##
        if index == -1:
            return self.dat[self.ncols]
        return self.dat[index]

    def __setitem__(self,int index, int value): ##CH##
        if index == -1:
            self.dat[self.ncols] = value
        self.dat[index] = value

##Current length of the array
    #def __len__(self): ##CH##
        #return self.icols

    cdef public int get(self,int index): ##CH##
        if index == -1:
            return self.dat[self.ncols]
        return self.dat[index]

    cdef public set(self,int index, int value): ##CH##
        if index == -1:
            self.dat[self.ncols] = value
        self.dat[index] = value

    cdef public add(self,int value): ##CH##
        cdef int value2
        if value >= self.ncols or value < -1:
            raise IndexError('add: value out of index')
        if value == -1:
            value2 = self.ncols
        else:
            value2 = value
        if self.dat_bool[value2] == -1:
            self.dat_bool[value2] = self.icols
            self.dat[self.icols] = value
            self.icols += 1

    cdef public remove(self,int value): ##CH##
        cdef int pos,i,value2
        if value >= self.ncols or value < -1:
            raise IndexError('remove: value out of index')
        if value == -1:
            value2 = self.ncols
        else:
            value2 = value
        pos = self.dat_bool[value2]
        if pos == -1:
            print 'value: ',value
            raise ValueError('remove: internal index error')
        self.dat_bool[value2] = -1
        for i in xrange(pos+1,self.icols):
            dati = self.dat[i]
            if dati == -1:
                dati = self.ncols
            self.dat_bool[dati] -= 1
            self.dat[i-1] = self.dat[i]
        self.icols -= 1

    def add_py(self,int value): ##CH##
        cdef int value2
        if value >= self.ncols or value < -1:
            raise IndexError('add: value out of index')
        if value == -1:
            value2 = self.ncols
        else:
            value2 = value
        if self.dat_bool[value2] == -1:
            self.dat_bool[value2] = self.icols
            self.dat[self.icols] = value
            self.icols += 1

    def remove_py(self,int value): ##CH##
        cdef int pos,i,value2
        if value >= self.ncols or value < -1:
            raise IndexError('remove: value out of index')
        if value == -1:
            value2 = self.ncols
        else:
            value2 = value
        pos = self.dat_bool[value2]
        self.dat_bool[value2] = -1
        for i in xrange(pos+1,self.icols):
            dati = self.dat[i]
            if dati == -1:
                dati = self.ncols
            self.dat_bool[dati] -= 1
            self.dat[i-1] = self.dat[i]
        self.icols -= 1

    cdef public Node_Set union(self,Node_Set l2):  ##CH##
        cdef int i
        cdef Node_Set lret
        if self.size() != l2.size():
            raise AssertionError('union: The size of the arrays must be equal')

        if l2.len() > self.len():
            lret = l2.copy()
            for i in xrange(self.icols):
                lret.add(self.dat[i])
        else:
            lret = self.copy()
            for i in xrange(l2.icols):
                lret.add(l2.dat[i])
        return lret

    cdef public Node_Set intersection(self,Node_Set l2): ##CH##
        cdef int i
        cdef Node_Set lret
        if self.size() != l2.size():
            raise AssertionError('Intersection: The size of the arrays must be equal')
        lret = Node_Set(self.size())

        if l2.len() > self.len():
            for i in xrange(self.icols):
                if l2.contains(self.dat[i]):
                    lret.add(self.dat[i])
        else:
            for i in xrange(l2.icols):
                if self.contains(l2.dat[i]):
                    lret.add(l2.dat[i])
        return lret

    cdef public Node_Set difference(self,Node_Set l2):
        cdef int i
        cdef Node_Set lret
        if self.size() != l2.size():
            raise AssertionError('Intersection: The size of the arrays must be equal')

        lret = Node_Set(self.size())

        for i in xrange(self.icols):
            if not l2.contains(self.dat[i]):
                lret.add(self.dat[i])
        return lret


    cdef public Node_Set copy(self): ##CH##
        cdef int i
        cdef Node_Set lret
        lret = Node_Set.__new__(Node_Set,self.size())
        # lret.name = self.name
        # lret.icols = self.icols
        # lret.ncols = self.ncols
        # for i in xrange(self.icols):
        #      lret.dat[i] = self.dat[i]
        #      lret.dat_bool[self.dat[i]] = i
        for i in xrange(self.icols):
            lret.add(self.dat[i])
        return lret

    cdef public bool contains(self,int index): ##CH##
        cdef int value2
        if index == -1:
            value2 = self.ncols
        else:
            value2 = index
        return self.dat_bool[value2] != -1


    #Current length of the array
    cdef public int len(self): ##CH##
        return self.icols

    #Current length of the array
    def len_py(self): ##CH##
        return self.icols

    # Size available for the array
    cdef public int size(self): ##CH##
        return self.ncols

    cdef public int pop(self): ##CH##
        if self.icols == 0:
            return -1
        self.icols -= 1
        if self.dat[self.icols] == -1:
            self.dat_bool[self.ncols] = -1
        else:
            self.dat_bool[self.dat[self.icols]] = -1
        return self.dat[self.icols]

    def display(self):  ##CH##
        cdef int i
        cdef list lst = []
        for i in xrange(self.icols):
            lst.append(self.dat[i])
        #print lst
        return lst

    def display_bool(self): ##CH##
        cdef int i
        cdef list lst = []
        for i in xrange(self.ncols+1):
            lst.append(self.dat_bool[i])
        #print lst
        return lst


    def clear(self): ##CH##
        cdef int i,xi
        for i in xrange(self.ncols + 1):
            self.dat_bool[i] = -1
        self.icols = 0
        
    def reverse(self):
        cdef int i,xi
        cdef Node_Set nodes_copy, nodes_rev
        
        nodes_rev = Node_Set.__new__(Node_Set,self.size())
        nodes_copy = self.copy()
        for i in range(self.icols):
            xi = nodes_copy.pop()
            nodes_rev.add(xi)
        return nodes_rev
        
        

