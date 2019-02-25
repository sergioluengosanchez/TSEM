#A data file
from libc.stdlib cimport malloc, free
from libc.math cimport log
from itertools import product
from time import time
cimport numpy
import numpy
import pandas
cdef extern from "log_likelihood.c":
    double logLik(const int* data, int nrows, int ncols, const int* vars, const int* ncat, int l_vars)

#Data used to compute the likelihood
cdef class data(object):
#    cdef int *dat_c
#    cdef readonly int nrows,ncols
#    cdef readonly list classes
#    cdef readonly list col_names

    def __init__(self,data_frame,classes=None):
        cdef int i
        cdef numpy.ndarray class_aux
        self.nrows,self.ncols = data_frame.shape
        self.dat_c = <int *>malloc(self.ncols * self.nrows * sizeof(int))
        self.col_names = list(data_frame)
        self.classes = classes
        if classes is None:
            self.classes = []
            for i in xrange(self.ncols):
               class_aux = data_frame.ix[:,i].unique()
               class_aux = class_aux[~pandas.isnull(class_aux)]
               self.classes = self.classes + [class_aux.tolist()]

        self.copy_dat_c(data_frame)

    def __dealloc__(self):
        cdef int i
        free(self.dat_c)

    cdef to_array(self, object data_frame):
        cdef int i,j
        cdef list classes
        cdef list col
        cdef int idx

        dat = numpy.zeros(data_frame.shape)

        for i in range(self.ncols):
            col = data_frame.ix[:,i].tolist()
            classes = self.classes[i]
            for j in range(self.nrows):
                if col[j] in classes:
                    idx =  classes.index(col[j])
                else:
                    idx = -1
                dat[j,i] = idx
        return dat

    cdef copy_dat_c(self, object data_frame):
        cdef int i,j,cont
        cdef list
        dat_array = self.to_array(data_frame)
        cont = 0
        for j in range(self.ncols):
            for i in range(self.nrows):
                self.dat_c[cont] = dat_array[i,j]
                cont = cont + 1
    
    def to_DataFrame(self):
        cdef int i,j,cont
        cdef list l_data, l_data_j
        cont = 0
        l_data = []
        for j in range(self.ncols):
            l_data_j = []
            for i in range(self.nrows):
                l_data_j.append(self.classes[j][self.dat_c[cont]])
                cont = cont + 1
            l_data.append(l_data_j)
        df = pandas.DataFrame(l_data)
        df = df.transpose()
        df.columns = self.col_names
        return df    
        
    
    def print_pos(self, i):
        print self.dat_c[i]

def print_l_py(data lst_in, int xi, int ln):
    cdef int* lst=&lst_in.dat_c[xi*ln]
    cdef list lst2
    lst2 = []
    for i in xrange(ln):
        lst2.append(lst[i])
    print lst2

cdef print_l(int* lst,int ln):
    cdef list lst2
    lst2 = []
    for i in xrange(ln):
        lst2.append(lst[i])
    print lst2
    

#log-likelihood
def ll_metric(int xi, list parents, data dat):
    cdef object vars_py
    cdef int *variables
    cdef int *ncat
    cdef int i
    cdef double score

    num_classes = [len(classi) for classi in dat.classes]
    # alloc variables and ncat
    vars_py = [xi] + parents
    variables = <int *>malloc(len(vars_py) * sizeof(int))
    ncat = <int *>malloc(len(num_classes) * sizeof(int))
    for i in range(len(vars_py)):
        variables[i] = vars_py[i]
    for i in range(len(num_classes)):
        ncat[i] = num_classes[i]
    try:
        score = logLik(dat.dat_c ,dat.nrows, dat.ncols, variables, ncat, len(vars_py))
    except:
        raise ValueError('Exception in logLik')
    if score > 0:
        print 'xi: ', xi, ', parents: ', parents, ', ncat: ', [num_classes[ni] for ni in [xi]+parents]
        print_l(&dat.dat_c[dat.nrows*xi], dat.nrows)
        for pi in parents:
            print_l(&dat.dat_c[dat.nrows*pi], dat.nrows)
        raise ValueError('Fail in logLik. Value out of range')
    return score

