#include <math.h>

void printDArray(const double* array, int len){
    int i;
    for ( i = 0; i < len; i++ ){
        printf("%f ",array[i]);
    }
}

void printIArray(const int* array, int len){
    int i;
    for ( i = 0; i < len; i++ ){
        printf("%i ",array[i]);
    }
}

/* Returns the position of the current data row in value to increment
 * its value */
int position(const int* d_row, const int* vars, const int* ncat, int l_vars, int nrows){
    int pos,times,i;
    pos = 0;
    times = 1;
    for ( i = 0; i < l_vars; i++ ){
        if (d_row[vars[i]*nrows]<= ncat[vars[i]]){
            pos += (d_row[vars[i]*nrows])*times;
            times *= ncat[vars[i]];
        } else{
            return -1;
        }
    }
    return pos;
}

/* Obtains the number of possible configurations of vars */
int nconf(const int* vars, const int* ncat, int l_vars){
    int i;
    int ncf = 1;
    for ( i = 0; i < l_vars; i++ ){
        ncf *= ncat[vars[i]];
    }
    return ncf;

}


/* Returns the frequencies of each possible configuration of vars */
int* freq(const int* data, int nrows, int ncols, const int* vars, const int* ncat, int l_vars){
    int *fq;
    int i,ncf,pos;

    ncf = nconf(vars,ncat,l_vars);
    fq = (int*) malloc(ncf * sizeof(int));
    /*Initialize fq*/
    for ( i = 0; i < ncf; i++ ){
        fq[i] = 0;
    }

    for ( i = 0; i < nrows; i++ ){
        pos = position(&(data[i]), vars, ncat, l_vars,nrows);
        if (pos == -1){
            return NULL;
        } else{
            fq[pos]++;
        }
    }

    return fq;
}

/* Returns ll for xi and paXi, xi must be the first element in vars*/
/* If error, it returns 1*/
double logLik(const int* data, int nrows, int ncols, const int* vars, const int* ncat, int l_vars){
    int *fq;
    int i,j,ncf,nc_xi,loops;
    double Nijk,Nij,ll = 0;
    /* Obtain frequencies */
    fq = freq(data, nrows, ncols, vars, ncat, l_vars);
    if (fq == NULL){
        return 1;
    }
    ncf = nconf(vars,ncat,l_vars);
    nc_xi = ncat[vars[0]];
    loops = ncf/nc_xi;

    for ( i = 0; i < loops; i++ ){
        /* Obtain Nij */
        Nij = 0;
        for ( j = 0; j < nc_xi; j++ ){
            Nij += fq[i*nc_xi + j];
        }
        if (Nij != 0) {
            /* Add to ll */
            for ( j = 0; j < nc_xi; j++ ){
                Nijk = fq[i*nc_xi + j];
                if (Nijk != 0){
                    ll += Nijk * log(Nijk/Nij);
                }
            }
        }
    }

    free(fq);
    return ll;
}
