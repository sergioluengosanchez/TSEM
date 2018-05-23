cdef class data(object):
    cdef int *dat_c
    cdef readonly int nrows,ncols
    cdef readonly list classes
    cdef readonly list col_names
    cdef to_array(self, object data_frame)
    cdef copy_dat_c(self, object data_frame)